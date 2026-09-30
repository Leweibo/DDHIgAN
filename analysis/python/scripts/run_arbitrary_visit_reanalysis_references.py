"""One-fit-per-fold Cox/RSF references for arbitrary-visit training."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from deephit.clinical_variable_sets import get_clinical_variable_set
from evaluation.evaluate_single_model import evaluate_landmark
from utils.data_utils import anchor_nearest_visit_at_t0, get_fold_split
from utils.reproducibility import effective_fold_seed
from utils.survival_data import split_patient_ids


EVALUATION_LANDMARKS = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
CURVE_TIMES = tuple(float(value) for value in range(1, 11))
HORIZONS = (3.0, 5.0, 10.0)


def _outcome_columns(outcome: str) -> tuple[str, str]:
    prefix = outcome.lower()
    return f"{prefix}_time", f"{prefix}_status"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(patient_ids: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(map(str, patient_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_formal_fit_ids(
    outer_train_ids: list[str], test_ids: list[str], model_name: str,
    outcomes: pd.DataFrame, status_col: str, fold: int,
) -> tuple[list[str], list[str]]:
    outer = list(map(str, outer_train_ids))
    test = set(map(str, test_ids))
    if set(outer).intersection(test):
        raise ValueError(f"fold {fold} outer-training and test IDs overlap")
    if model_name == "baseline_cox":
        return outer, []
    strata = (
        outcomes.assign(patient_id=outcomes["patient_id"].astype(str))
        .set_index("patient_id")[status_col]
        .reindex(outer).fillna(-1).astype(int).tolist()
    )
    return split_patient_ids(outer, 0.2, effective_fold_seed(316, fold), strata)


def _anchored_longitudinal(longitudinal: pd.DataFrame) -> pd.DataFrame:
    frame = longitudinal.copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    frame["visit_time"] = pd.to_numeric(frame["visit_time"], errors="coerce")
    frame = frame[np.isfinite(frame["visit_time"])].copy()
    return pd.concat(
        [anchor_nearest_visit_at_t0(group, 0.0) for _, group in frame.groupby("patient_id", sort=False)],
        ignore_index=True,
    )


def _base_frame(
    baseline: pd.DataFrame, outcomes: pd.DataFrame, patient_ids: list[str], outcome: str,
) -> pd.DataFrame:
    time_col, status_col = _outcome_columns(outcome)
    base = baseline.copy()
    base["patient_id"] = base["patient_id"].astype(str)
    out = outcomes[["patient_id", time_col, status_col]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[status_col] = pd.to_numeric(out[status_col], errors="coerce")
    order = pd.DataFrame({"patient_id": list(map(str, patient_ids))})
    result = order.merge(base, on="patient_id", how="inner", validate="one_to_one")
    result = result.merge(out, on="patient_id", how="inner", validate="one_to_one")
    return result[np.isfinite(result[time_col]) & result[status_col].isin([0, 1])].copy()


def _fill_dynamic(
    frame: pd.DataFrame, dynamic_cols: tuple[str, ...], fallback: dict[str, str],
) -> pd.DataFrame:
    result = frame.copy()
    for column in dynamic_cols:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[fallback[column]] = pd.to_numeric(result[fallback[column]], errors="coerce")
        result[column] = result[column].fillna(result[fallback[column]])
    return result


def build_actual_visit_training_frame(
    baseline: pd.DataFrame, longitudinal: pd.DataFrame, outcomes: pd.DataFrame,
    patient_ids: list[str], outcome: str = "ESKD", max_query_time: float = 5.0,
) -> pd.DataFrame:
    """Return real 0--5 year visit rows with variable-wise last values and patient weights."""
    variables = get_clinical_variable_set("minimal_core")
    base = _base_frame(baseline, outcomes, patient_ids, outcome)
    time_col, status_col = _outcome_columns(outcome)
    long = _anchored_longitudinal(longitudinal)
    for column in variables.dynamic_cols:
        long[column] = pd.to_numeric(long[column], errors="coerce")
        long[column] = long.groupby("patient_id", sort=False)[column].ffill()
    queries = long[
        (long["visit_time"] >= 0.0) & (long["visit_time"] <= float(max_query_time))
    ][["patient_id", "visit_time", *variables.dynamic_cols]].copy()
    result = queries.merge(base, on="patient_id", how="inner", validate="many_to_one")
    result = result[result["visit_time"] < result[time_col]].copy()
    result = _fill_dynamic(result, variables.dynamic_cols, variables.baseline_fallback)
    result["query_time"] = result["visit_time"].astype(float)
    result["duration"] = result[time_col].astype(float) - result["query_time"]
    result["event"] = result[status_col].astype(int)
    counts = result.groupby("patient_id")["patient_id"].transform("size").astype(float)
    result["patient_weight"] = 1.0 / counts
    columns = [
        "patient_id", "duration", "event", "patient_weight", "query_time",
        *variables.cox_static_cols, *variables.dynamic_cols,
    ]
    return result[columns].reset_index(drop=True)


def build_pccox_actual_visit_training_frame(
    baseline: pd.DataFrame, longitudinal: pd.DataFrame, outcomes: pd.DataFrame,
    patient_ids: list[str], outcome: str = "ESKD", max_query_time: float = 5.0,
) -> pd.DataFrame:
    """Return the unweighted actual-visit input required by ``PC.Cox``.

    ``stime`` remains biopsy-anchored absolute follow-up.  The package itself
    constructs residual survival as ``stime - measurement_time``.  Keeping
    this adapter separate prevents the retired visit-Cox patient weights from
    leaking into PCCox.
    """
    frame = build_actual_visit_training_frame(
        baseline, longitudinal, outcomes, patient_ids, outcome, max_query_time,
    ).drop(columns="patient_weight")
    time_col, _ = _outcome_columns(outcome)
    absolute = _base_frame(baseline, outcomes, patient_ids, outcome)[
        ["patient_id", time_col]
    ]
    frame = frame.merge(absolute, on="patient_id", how="left", validate="many_to_one")
    frame = frame.rename(columns={
        time_col: "stime", "event": "status", "query_time": "measurement_time",
    }).drop(columns="duration")
    return frame[["patient_id", "stime", "status", "measurement_time",
                  *get_clinical_variable_set("minimal_core").cox_static_cols,
                  *get_clinical_variable_set("minimal_core").dynamic_cols]]


def build_landmark_evaluation_frame(
    baseline: pd.DataFrame, longitudinal: pd.DataFrame, outcomes: pd.DataFrame,
    patient_ids: list[str], landmark: float, outcome: str = "ESKD",
) -> pd.DataFrame:
    variables = get_clinical_variable_set("minimal_core")
    base = _base_frame(baseline, outcomes, patient_ids, outcome)
    time_col, status_col = _outcome_columns(outcome)
    base = base[base[time_col] > float(landmark)].copy()
    long = _anchored_longitudinal(longitudinal)
    long = long[long["visit_time"] <= float(landmark)].copy()
    for column in variables.dynamic_cols:
        long[column] = pd.to_numeric(long[column], errors="coerce")
        long[column] = long.groupby("patient_id", sort=False)[column].ffill()
    latest = long.groupby("patient_id", as_index=False, sort=False).tail(1)[
        ["patient_id", *variables.dynamic_cols]
    ]
    result = base.merge(latest, on="patient_id", how="left", validate="one_to_one")
    result = _fill_dynamic(result, variables.dynamic_cols, variables.baseline_fallback)
    result["query_time"] = float(landmark)
    result["duration"] = result[time_col].astype(float) - float(landmark)
    result["event"] = result[status_col].astype(int)
    columns = [
        "patient_id", "duration", "event", "query_time",
        *variables.cox_static_cols, *variables.dynamic_cols,
    ]
    return result[columns].reset_index(drop=True)


def impute_from_training(
    train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_result, test_result = train.copy(), test.copy()
    for column in feature_cols:
        train_result[column] = pd.to_numeric(train_result[column], errors="coerce")
        test_result[column] = pd.to_numeric(test_result[column], errors="coerce")
        mean = float(train_result[column].mean())
        if not np.isfinite(mean):
            mean = 0.0
        train_result[column] = train_result[column].fillna(mean)
        test_result[column] = test_result[column].fillna(mean)
    return train_result, test_result


def _survival_at(survival: pd.DataFrame, time_value: float) -> pd.Series:
    index = np.asarray(survival.index, dtype=float)
    position = int(np.argmin(np.abs(index - float(time_value))))
    if not np.isclose(index[position], time_value, atol=1e-8):
        raise ValueError(f"survival table does not contain time {time_value:g}")
    return survival.iloc[position].astype(float)


def make_predictions(
    test: pd.DataFrame, landmark: float, survival: pd.DataFrame, *, absolute_time: bool,
) -> pd.DataFrame:
    denominator = _survival_at(survival, landmark if absolute_time else 0.0).clip(lower=1e-12)
    result = pd.DataFrame({
        "patient_id": test["patient_id"].astype(str).to_numpy(),
        "true_time": test["duration"].to_numpy(float) + landmark,
        "residual_time": test["duration"].to_numpy(float),
        "true_event": test["event"].to_numpy(int),
        "Tstart": float(landmark),
    })
    for horizon in CURVE_TIMES:
        time_value = landmark + horizon if absolute_time else horizon
        conditional = (_survival_at(survival, time_value) / denominator).clip(0.0, 1.0)
        result[f"resid_surv_{horizon:.1f}y"] = conditional.to_numpy()
        if horizon in HORIZONS:
            result[f"cond_surv_{int(horizon)}y"] = conditional.to_numpy()
    result["risk_score"] = 1.0 - result["cond_surv_10y"]
    return result


def _validation_frame(
    frame: pd.DataFrame, survival: np.ndarray,
) -> pd.DataFrame:
    result = frame[["patient_id", "query_time", "duration", "event", "patient_weight"]].rename(
        columns={"duration": "residual_time", "event": "true_event", "patient_weight": "sample_weight"}
    ).reset_index(drop=True)
    for index, horizon in enumerate(CURVE_TIMES):
        result[f"resid_surv_{horizon:.1f}y"] = survival[index]
    return result


def _baseline_validation_survival(model, features: pd.DataFrame, queries: np.ndarray) -> np.ndarray:
    hazard = model.baseline_cumulative_hazard_.iloc[:, 0]
    times = np.asarray(hazard.index, dtype=float)
    values = hazard.to_numpy(float)

    def at(query):
        index = np.searchsorted(times, query, side="right") - 1
        return np.where(index < 0, 0.0, values[np.maximum(index, 0)])

    partial_hazard = model.predict_partial_hazard(features).to_numpy(float)
    return np.asarray([
        np.exp(-(at(queries + horizon) - at(queries)) * partial_hazard)
        for horizon in CURVE_TIMES
    ])


def require_rsf_sample_weight_support() -> None:
    from sksurv.ensemble import RandomSurvivalForest

    if "sample_weight" not in inspect.signature(RandomSurvivalForest.fit).parameters:
        raise RuntimeError("target RandomSurvivalForest.fit does not support sample_weight")


def _write_outputs(
    predictions: pd.DataFrame, train_evaluation: pd.DataFrame, output_dir: Path,
    prefix: str, outcome: str, fold: int, landmark: float,
) -> dict:
    stem = f"{prefix}_{outcome}_fold{fold}_test_landmark{int(landmark)}"
    prediction_path = output_dir / f"{stem}.csv"
    metric_path = output_dir / f"{prefix}_{outcome}_fold{fold}_landmark{int(landmark)}_ipcw_metrics.json"
    calibration_path = output_dir / f"{prefix}_{outcome}_fold{fold}_landmark{int(landmark)}_calibration.csv"
    predictions.to_csv(prediction_path, index=False)
    metrics, calibration = evaluate_landmark(
        predictions, train_time=train_evaluation["duration"].to_numpy(float),
        train_event=train_evaluation["event"].to_numpy(int), landmark=landmark,
        horizons=list(HORIZONS), max_horizon=10.0,
    )
    metric_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    calibration.to_csv(calibration_path, index=False)
    return metrics


def run_fold(
    baseline: pd.DataFrame, longitudinal: pd.DataFrame, outcomes: pd.DataFrame,
    splits_dir: Path, output_dir: Path, fold: int, outcome: str, model_name: str,
    penalizer: float, n_jobs: int,
    evaluation_landmarks: tuple[float, ...] = EVALUATION_LANDMARKS,
) -> list[dict]:
    from lifelines import CoxPHFitter
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv

    variables = get_clinical_variable_set("minimal_core")
    feature_cols = [*variables.cox_static_cols, *variables.dynamic_cols]
    outer_train_ids = get_fold_split(splits_dir, fold, "train")
    test_ids = get_fold_split(splits_dir, fold, "test")
    time_col, status_col = _outcome_columns(outcome)
    train_ids, validation_ids = select_formal_fit_ids(
        outer_train_ids, test_ids, model_name, outcomes, status_col, fold,
    )
    if model_name == "baseline_cox":
        # Baseline Cox has no algorithmic validation/early-stopping step.  Its
        # formal fair-comparison fit therefore uses every outer-training ID.
        visit_train = validation = None
    else:
        visit_train = build_actual_visit_training_frame(
            baseline, longitudinal, outcomes, train_ids, outcome,
        )
        validation = build_actual_visit_training_frame(
            baseline, longitudinal, outcomes, validation_ids, outcome,
        )
    rows = []
    if model_name == "visit_cox":
        model = CoxPHFitter(penalizer=penalizer)
        formula = " + ".join(feature_cols) + " + bs(query_time, df=3, degree=3)"
        fit_frame, _ = impute_from_training(visit_train, visit_train, feature_cols)
        model.fit(
            fit_frame[["patient_id", "duration", "event", "patient_weight", "query_time", *feature_cols]],
            duration_col="duration", event_col="event", weights_col="patient_weight",
            cluster_col="patient_id", robust=True, formula=formula, show_progress=False,
        )
        prefix = "visit_updated_cox_minimal_core"
        _, validation_imputed = impute_from_training(visit_train, validation, feature_cols)
        validation_features = validation_imputed[["query_time", *feature_cols]].copy()
        validation_survival = model.predict_survival_function(
            validation_features, times=list(CURVE_TIMES)
        ).to_numpy(float)
    elif model_name == "visit_rsf":
        require_rsf_sample_weight_support()
        fit_frame, _ = impute_from_training(visit_train, visit_train, [*feature_cols, "query_time"])
        model = RandomSurvivalForest(
            n_estimators=500, min_samples_split=10, min_samples_leaf=5,
            max_features="sqrt", n_jobs=n_jobs, random_state=316 + fold,
        )
        model.fit(
            fit_frame[[*feature_cols, "query_time"]],
            Surv.from_arrays(fit_frame["event"].astype(bool), fit_frame["duration"].astype(float)),
            sample_weight=fit_frame["patient_weight"].to_numpy(float),
        )
        prefix = "visit_updated_rsf_minimal_core"
        _, validation_imputed = impute_from_training(
            visit_train, validation, [*feature_cols, "query_time"]
        )
        functions = model.predict_survival_function(
            validation_imputed[[*feature_cols, "query_time"]], return_array=False,
        )
        validation_survival = np.empty((len(CURVE_TIMES), len(validation_imputed)))
        for column, function in enumerate(functions):
            x, y = np.asarray(function.x), np.asarray(function.y)
            for row, time_value in enumerate(CURVE_TIMES):
                index = np.searchsorted(x, time_value, side="right") - 1
                validation_survival[row, column] = 1.0 if index < 0 else y[index]
    else:
        baseline_train = build_landmark_evaluation_frame(
            baseline, longitudinal, outcomes, train_ids, 0.0, outcome,
        )
        baseline_features = [
            *variables.cox_static_cols,
            *[variables.baseline_fallback[column] for column in variables.dynamic_cols],
        ]
        base = _base_frame(baseline, outcomes, train_ids, outcome)
        time_col, status_col = _outcome_columns(outcome)
        base["duration"] = base[time_col].astype(float)
        base["event"] = base[status_col].astype(int)
        base, _ = impute_from_training(base, base, baseline_features)
        model = CoxPHFitter(penalizer=penalizer)
        model.fit(base[["duration", "event", *baseline_features]], "duration", "event", show_progress=False)
        fitted_ids = base["patient_id"].astype(str).tolist()
        if set(fitted_ids) != set(map(str, outer_train_ids)) or len(fitted_ids) != len(outer_train_ids):
            raise ValueError(f"fold {fold} Baseline Cox fit IDs differ from outer-training IDs")
        if set(fitted_ids).intersection(map(str, test_ids)):
            raise ValueError(f"fold {fold} Baseline Cox fit IDs overlap the test fold")
        prefix = "baseline_cox_minimal_core_full_outer_train"

    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name != "baseline_cox":
        _validation_frame(validation, validation_survival).to_csv(
            output_dir / f"{prefix}_{outcome}_fold{fold}_inner_validation_actual_queries.csv",
            index=False,
        )
    else:
        manifest = {
            "status": "READY",
            "model": "Baseline Cox",
            "artifact_prefix": prefix,
            "fold": fold,
            "outer_train_patients": len(outer_train_ids),
            "fit_patients": len(train_ids),
            "excluded_outer_train_patients": 0,
            "test_patients": len(test_ids),
            "fit_test_overlap": 0,
            "outer_train_ids_sha256": _ids_sha256(outer_train_ids),
            "fit_ids_sha256": _ids_sha256(train_ids),
            "test_ids_sha256": _ids_sha256(test_ids),
            "split_train_sha256": _sha256(splits_dir / f"fold_{fold}_train.csv"),
            "split_test_sha256": _sha256(splits_dir / f"fold_{fold}_test.csv"),
            "penalizer": penalizer,
            "raw_predictions": True,
            "inner_validation_used": False,
        }
        (output_dir / f"{prefix}_{outcome}_fold{fold}_fit_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
    for landmark in evaluation_landmarks:
        train_eval = build_landmark_evaluation_frame(
            baseline, longitudinal, outcomes, outer_train_ids, landmark, outcome,
        )
        test = build_landmark_evaluation_frame(
            baseline, longitudinal, outcomes, test_ids, landmark, outcome,
        )
        if model_name == "visit_cox":
            _, test = impute_from_training(visit_train, test, feature_cols)
            features = test[["query_time", *feature_cols]].copy()
            features.index = test["patient_id"].astype(str)
            times = [0.0, *CURVE_TIMES]
            survival = model.predict_survival_function(features, times=times)
            predictions = make_predictions(test, landmark, survival, absolute_time=False)
        elif model_name == "visit_rsf":
            _, test = impute_from_training(visit_train, test, [*feature_cols, "query_time"])
            functions = model.predict_survival_function(test[[*feature_cols, "query_time"]], return_array=False)
            survival = pd.DataFrame(index=[0.0, *CURVE_TIMES], columns=range(len(test)), dtype=float)
            survival.loc[0.0] = 1.0
            for column, function in enumerate(functions):
                x, y = np.asarray(function.x), np.asarray(function.y)
                for time_value in CURVE_TIMES:
                    index = np.searchsorted(x, time_value, side="right") - 1
                    survival.loc[time_value, column] = 1.0 if index < 0 else y[index]
            predictions = make_predictions(test, landmark, survival, absolute_time=False)
        else:
            baseline_features = [
                *variables.cox_static_cols,
                *[variables.baseline_fallback[column] for column in variables.dynamic_cols],
            ]
            base_train = _base_frame(baseline, outcomes, outer_train_ids, outcome)
            test_base = _base_frame(baseline, outcomes, test["patient_id"].tolist(), outcome)
            _, test_base = impute_from_training(base_train, test_base, baseline_features)
            features = test_base.set_index("patient_id")[baseline_features]
            times = sorted({landmark, *[landmark + value for value in CURVE_TIMES]})
            survival = model.predict_survival_function(features, times=times)
            predictions = make_predictions(test, landmark, survival, absolute_time=True)
        metrics = _write_outputs(
            predictions, train_eval, output_dir, prefix, outcome, fold, landmark,
        )
        rows.append({"model": model_name, "fold": fold, "landmark": landmark, "fit_count": 1, **metrics})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("baseline_cox", "visit_cox", "visit_rsf"))
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--landmarks", default=",".join(str(int(value)) for value in EVALUATION_LANDMARKS))
    parser.add_argument("--outcome", default="ESKD", choices=("ESKD", "Drop50"))
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--preflight-rsf-sample-weight", action="store_true")
    args = parser.parse_args()
    if args.preflight_rsf_sample_weight:
        require_rsf_sample_weight_support()
        print("RSF_SAMPLE_WEIGHT_OK")
        return 0
    if not args.model:
        parser.error("--model is required unless --preflight-rsf-sample-weight is used")
    baseline = pd.read_csv(args.processed_dir / "igan_baseline.csv", low_memory=False)
    longitudinal = pd.read_csv(args.processed_dir / "igan_longitudinal.csv", low_memory=False)
    outcomes = pd.read_csv(args.processed_dir / "igan_outcomes.csv", low_memory=False)
    rows = []
    folds = [int(value) for value in args.folds.split(",")]
    landmarks = tuple(float(value) for value in args.landmarks.split(","))
    if not landmarks or len(set(landmarks)) != len(landmarks) or any(value < 0 for value in landmarks):
        raise ValueError("evaluation landmarks must be unique nonnegative values")
    for fold in folds:
        rows.extend(run_fold(
            baseline, longitudinal, outcomes, args.splits_dir, args.output_dir,
            fold, args.outcome, args.model, args.penalizer, args.n_jobs, landmarks,
        ))
    pd.DataFrame(rows).to_csv(args.output_dir / f"{args.model}_{args.outcome}_metrics_summary.csv", index=False)
    if args.model == "baseline_cox":
        prefix = "baseline_cox_minimal_core_full_outer_train"
        run_manifest = {
            "status": "COMPLETE", "model": "Baseline Cox", "artifact_prefix": prefix,
            "folds": folds, "fit_count_per_fold": 1, "penalizer": args.penalizer,
            "evaluation_landmarks": list(landmarks),
            "full_outer_training_required": True, "inner_validation_used": False,
            "raw_predictions": True,
            "source_input_sha256": {
                name: _sha256(args.processed_dir / name)
                for name in ("igan_baseline.csv", "igan_longitudinal.csv", "igan_outcomes.csv")
            },
            "fold_manifest_sha256": _sha256(args.splits_dir / "fold_manifest.json"),
            "runner_sha256": _sha256(Path(__file__)),
        }
        (args.output_dir / f"{prefix}_{args.outcome}_run_manifest.json").write_text(
            json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
