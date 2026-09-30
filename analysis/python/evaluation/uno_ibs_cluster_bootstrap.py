"""Aggregate-only physv11 evaluation from frozen outer-fold OOF predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd

from python.evaluation.paired_cluster_bootstrap import (
    _censoring_step,
    _step_query,
    _validate_arrays,
    canonicalize,
)
from python.evaluation.evaluate_single_model import make_landmark_outcomes
from python.utils.metrics import calibration_groups
from python.utils.probabilities import project_probability_roundoff


MODELS = {
    "DDHIgAN": ("clinical_ddh/predictions", "ddhigan_minimal_physv11_expanded_ge1y_20260813"),
    "DDHIgAN-mini": ("../ablation_physv11_no_cystatinc_20260815/predictions", "ddhigan_mini_no_cystatinc_physv11_expanded_ge1y_20260813"),
    "Baseline Cox": ("cox/baseline", "baseline_cox_minimal_core_strict"),
    "Landmark Cox": ("cox/landmark", "landmark_cox_minimal_core_strict"),
    "Landmark RSF": ("rsf", "rsf_minimal_core_strict"),
    "DDHIgAN-Pathology": ("pathology_ddh/predictions", "ddhigan_pathology_physv11_expanded_ge1y_20260813"),
    "DDHIgAN-Expanded": ("expanded_ddh/predictions", "ddhigan_expanded_physv11_expanded_ge1y_20260813"),
}
ALLHISTORY_MODEL = "DDHIgAN-AllHistory"
LANDMARKS = (0, 1, 3, 5)
FOLDS = tuple(range(5))
IBS_NODES = (1.0, 2.0, 3.0, 5.0, 7.0, 10.0)
METRICS = (
    "uno_c_index",
    "ipcw_ibs_1_10y",
    "auc_3y",
    "auc_5y",
    "auc_10y",
    "calibration_mae_3y",
    "calibration_mae_5y",
    "calibration_mae_10y",
    "brier_1y", "brier_2y", "brier_3y", "brier_5y", "brier_7y", "brier_10y",
    "calibration_intercept_5y", "calibration_slope_5y",
    "calibration_intercept_10y", "calibration_slope_10y",
)

_DATA = None


def parse_landmarks(value: str) -> tuple[int, ...]:
    landmarks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not landmarks or len(set(landmarks)) != len(landmarks):
        raise ValueError("landmarks must be a non-empty unique comma-separated list")
    return landmarks


def select_models(names: str | None = None, allhistory_directory: str | None = None,
                  allhistory_prefix: str | None = None) -> dict:
    if bool(allhistory_directory) != bool(allhistory_prefix):
        raise ValueError("AllHistory directory and prefix must be provided together")
    models = dict(MODELS)
    if allhistory_directory:
        models[ALLHISTORY_MODEL] = (allhistory_directory, allhistory_prefix)
    selected = list(models) if names is None else [value.strip() for value in names.split(",") if value.strip()]
    unknown = set(selected) - set(models)
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    if not selected or selected[0] != "DDHIgAN" or len(set(selected)) != len(selected):
        raise ValueError("models must be unique and start with DDHIgAN")
    return {name: models[name] for name in selected}


def parse_model_specs(specs: list[str]) -> dict:
    models = {}
    for spec in specs:
        try:
            name, directory, prefix = spec.split("=", 2)
        except ValueError as error:
            raise ValueError("model specs must be NAME=DIRECTORY=PREFIX") from error
        if not name or not directory or not prefix or name in models:
            raise ValueError("model specs require unique non-empty fields")
        models[name] = (directory, prefix)
    if not models or next(iter(models)) != "DDHIgAN":
        raise ValueError("custom models must start with DDHIgAN")
    return models


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _uno_c_index(time, event, risk, censoring, tau=10.0):
    time, event = _validate_arrays(time, event)
    risk = np.asarray(risk, dtype=float)
    unique_risk, ranks = np.unique(risk, return_inverse=True)
    tree = np.zeros(len(unique_risk) + 1, dtype=np.int64)

    def update(rank):
        index = int(rank) + 1
        while index < len(tree):
            tree[index] += 1
            index += index & -index

    def query(rank):
        total = 0
        index = int(rank) + 1
        while index > 0:
            total += int(tree[index])
            index -= index & -index
        return total

    order = np.argsort(-time, kind="stable")
    concordant = comparable = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        group_time = time[order[start]]
        while stop < len(order) and time[order[stop]] == group_time:
            stop += 1
        later_count = query(len(unique_risk) - 1)
        if group_time <= tau and later_count:
            group = order[start:stop]
            events = group[event[group] == 1]
            if len(events):
                g = _step_query(*censoring, time[events], before=True)
                for row, weight in zip(events, 1.0 / (g * g)):
                    lower = query(ranks[row] - 1)
                    equal = query(ranks[row]) - lower
                    concordant += float(weight) * (lower + 0.5 * equal)
                    comparable += float(weight) * later_count
        for row in order[start:stop]:
            update(ranks[row])
        start = stop
    if comparable == 0:
        raise ValueError("no comparable pairs for Uno C-index")
    return concordant / comparable


def _auc(time, event, risk, censoring, horizon):
    time, event = _validate_arrays(time, event)
    risk = np.asarray(risk, dtype=float)
    cases = np.flatnonzero((time <= horizon) & (event == 1))
    controls = np.flatnonzero(time > horizon)
    if not len(cases) or not len(controls):
        raise ValueError("dynamic AUC requires cases and controls")
    weights = 1.0 / _step_query(*censoring, time[cases], before=True)
    control_risk = np.sort(risk[controls])
    lower = np.searchsorted(control_risk, risk[cases], side="left")
    upper = np.searchsorted(control_risk, risk[cases], side="right")
    return float(np.sum(weights * (lower + 0.5 * (upper - lower))) / (weights.sum() * len(controls)))


def _ibs(time, event, predicted_survival, censoring):
    time, event = _validate_arrays(time, event)
    scores = []
    for column, horizon in enumerate(IBS_NODES):
        event_before = (time <= horizon) & (event == 1)
        event_free = time > horizon
        weights = np.zeros(len(time), dtype=float)
        weights[event_before] = 1.0 / _step_query(*censoring, time[event_before], before=True)
        if event_free.any():
            weights[event_free] = 1.0 / float(_step_query(*censoring, np.asarray(horizon)))
        scores.append(np.mean(weights * (event_free.astype(float) - predicted_survival[:, column]) ** 2))
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(scores, IBS_NODES) / 9.0)


def _metrics(frame: pd.DataFrame, censoring) -> np.ndarray:
    time = frame["residual_time"].to_numpy(float)
    event = frame["true_event"].to_numpy(int)
    risks = {
        horizon: 1.0 - frame[f"cond_surv_{horizon}y"].to_numpy(float)
        for horizon in (3, 5, 10)
    }
    survival = frame[[f"resid_surv_{time:.1f}y" for time in IBS_NODES]].to_numpy(float)
    brier = []
    for column, horizon in enumerate(IBS_NODES):
        event_before = (time <= horizon) & (event == 1)
        event_free = time > horizon
        weights = np.zeros(len(time), dtype=float)
        if event_before.any():
            weights[event_before] = 1.0 / _step_query(
                *censoring, time[event_before], before=True,
            )
        if event_free.any():
            weights[event_free] = 1.0 / float(_step_query(*censoring, np.asarray(horizon)))
        brier.append(float(np.mean(weights * (event_free.astype(float) - survival[:, column]) ** 2)))
    values = [
        _uno_c_index(time, event, risks[10], censoring, tau=10.0),
        _ibs(time, event, survival, censoring),
        *(_auc(time, event, risks[horizon], censoring, float(horizon)) for horizon in (3, 5, 10)),
    ]
    for horizon in (3, 5, 10):
        risk = risks[horizon]
        groups = calibration_groups(time, event, risk, horizon, n_groups=5)
        values.append(float(np.mean([abs(row["mean_predicted_risk"] - row["observed_risk"]) for row in groups])))
    values.extend(brier)
    for horizon in (5, 10):
        risk = np.clip(risks[horizon], 1e-6, 1 - 1e-6)
        cases = (time <= horizon) & (event == 1)
        controls = time > horizon
        usable = cases | controls
        weights = np.zeros(len(time), dtype=float)
        weights[cases] = 1.0 / _step_query(*censoring, time[cases], before=True)
        weights[controls] = 1.0 / float(_step_query(*censoring, np.asarray(float(horizon))))
        design = np.column_stack([np.ones(int(usable.sum())), np.log(risk[usable] / (1 - risk[usable]))])
        outcome = cases[usable].astype(float)
        beta = np.asarray([0.0, 1.0])
        for _ in range(50):
            fitted = 1.0 / (1.0 + np.exp(-np.clip(design @ beta, -30, 30)))
            score = design.T @ (weights[usable] * (outcome - fitted))
            information = design.T @ ((weights[usable] * fitted * (1 - fitted))[:, None] * design)
            step = np.linalg.pinv(information) @ score
            beta += step
            if np.max(np.abs(step)) < 1e-8:
                break
        values.extend(beta.tolist())
    return np.asarray(values)


def _prediction_path(root: Path, model: str, fold: int, landmark: int, models=MODELS) -> Path:
    directory, prefix = models[model]
    return root / directory / f"{prefix}_ESKD_fold{fold}_test_landmark{landmark}.csv"


def load_data(prediction_root: Path, processed_dir: Path, splits_dir: Path,
              models=MODELS, landmarks=LANDMARKS):
    outcomes = pd.read_csv(processed_dir / "igan_outcomes.csv", dtype={"patient_id": str})
    cells = {}
    reference_keys = {}
    prediction_paths = []
    all_ids = set()
    for fold in FOLDS:
        train_ids = set(pd.read_csv(splits_dir / f"fold_{fold}_train.csv", dtype={"patient_id": str})["patient_id"])
        for landmark in landmarks:
            train = make_landmark_outcomes(outcomes[outcomes["patient_id"].isin(train_ids)], "ESKD", landmark)
            censoring = _censoring_step(train["residual_time"].to_numpy(float), train["event"].to_numpy(int))
            for model in models:
                path = _prediction_path(prediction_root, model, fold, landmark, models)
                prediction_paths.append(path)
                frame = pd.read_csv(path, dtype={"patient_id": str}).sort_values("patient_id").reset_index(drop=True)
                required = {"patient_id", "residual_time", "true_event", "cond_surv_5y", "cond_surv_10y"}
                required.update(f"resid_surv_{time:.1f}y" for time in IBS_NODES)
                missing = required - set(frame)
                if missing:
                    raise ValueError(f"{path} missing {sorted(missing)}")
                if frame["patient_id"].duplicated().any():
                    raise ValueError(f"duplicate patient in {path}")
                probability_columns = [
                    column for column in frame.columns
                    if column == "risk_score" or column.startswith("cond_surv_") or column.startswith("resid_surv_")
                ]
                projected, probability_audit = project_probability_roundoff(
                    frame[probability_columns].to_numpy(dtype=float)
                )
                frame.loc[:, probability_columns] = projected
                frame.attrs["probability_roundoff"] = probability_audit
                key = (fold, landmark)
                identity = tuple(zip(frame["patient_id"], frame["true_event"]))
                residual = frame["residual_time"].to_numpy(dtype=float)
                if key in reference_keys:
                    reference_identity, reference_residual = reference_keys[key]
                    # patient_id/true_event must match exactly; residual_time tolerates
                    # 1e-9 because R-written predictions round the last printed digit.
                    if identity != reference_identity or not np.allclose(
                        residual, reference_residual, rtol=0.0, atol=1e-9
                    ):
                        raise ValueError(f"model risk set or labels differ at fold {fold}, landmark {landmark}")
                reference_keys.setdefault(key, (identity, residual))
                cells[(model, fold, landmark)] = (frame, censoring)
                all_ids.update(frame["patient_id"])
    patient_ids = np.asarray(sorted(all_ids), dtype=object)
    id_index = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    for model in models:
        for fold in FOLDS:
            for landmark in landmarks:
                frame, censoring = cells[(model, fold, landmark)]
                cells[(model, fold, landmark)] = (
                    frame,
                    censoring,
                    np.fromiter((id_index[value] for value in frame["patient_id"]), dtype=int),
                )
    return cells, patient_ids, prediction_paths


def _evaluate_counts(counts: np.ndarray, models=MODELS, landmarks=LANDMARKS,
                     summary_landmarks=LANDMARKS) -> np.ndarray:
    result = np.empty((len(models), len(landmarks) + 1, len(METRICS)))
    for model_index, model in enumerate(models):
        metric_values = []
        for landmark_index, landmark in enumerate(landmarks):
            folds = []
            for fold in FOLDS:
                frame, censoring, global_indices = _DATA[(model, fold, landmark)]
                selected = np.repeat(np.arange(len(frame)), counts[global_indices])
                if not len(selected):
                    raise ValueError("bootstrap draw contains an empty fold-landmark risk set")
                folds.append(_metrics(frame.iloc[selected], censoring))
            value = np.mean(folds, axis=0)
            result[model_index, landmark_index] = value
            metric_values.append(value)
        summary_indices = [landmarks.index(value) for value in summary_landmarks]
        result[model_index, -1] = np.mean(
            [metric_values[index] for index in summary_indices], axis=0
        )
    return result


def _init_worker(data):
    global _DATA
    _DATA = data


def _summary(estimate: float, draws: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "bootstrap_mean": float(np.mean(draws)),
        "ci_lower": float(np.quantile(draws, 0.025)),
        "ci_upper": float(np.quantile(draws, 0.975)),
    }


def _calibration_curves(cells, models=MODELS, landmarks=LANDMARKS) -> list[dict]:
    rows = []
    for model in models:
        for landmark in landmarks:
            by_horizon = {5: [], 10: []}
            for fold in FOLDS:
                frame = cells[(model, fold, landmark)][0]
                time = frame["residual_time"].to_numpy(float)
                event = frame["true_event"].to_numpy(int)
                for horizon in by_horizon:
                    risk = 1.0 - frame[f"cond_surv_{horizon}y"].to_numpy(float)
                    by_horizon[horizon].append(calibration_groups(time, event, risk, horizon, n_groups=5))
            for horizon, fold_groups in by_horizon.items():
                for group_index in range(5):
                    group_rows = [fold[group_index] for fold in fold_groups]
                    n = sum(row["n"] for row in group_rows)
                    rows.append({
                        "model": model,
                        "landmark_years": landmark,
                        "horizon_years": horizon,
                        "group": group_index + 1,
                        "n": n,
                        "mean_predicted_risk": sum(row["n"] * row["mean_predicted_risk"] for row in group_rows) / n,
                        "observed_risk": sum(row["n"] * row["observed_risk"] for row in group_rows) / n,
                    })
    return rows


def _smoothed_calibration_curves(cells, models=MODELS, landmarks=LANDMARKS) -> list[dict]:
    rows = []
    for model in models:
        for landmark in landmarks:
            for horizon in (5, 10):
                risks, outcomes, weights = [], [], []
                for fold in FOLDS:
                    frame, censoring = cells[(model, fold, landmark)][:2]
                    time = frame["residual_time"].to_numpy(float)
                    event = frame["true_event"].to_numpy(int)
                    risk = 1.0 - frame[f"cond_surv_{horizon}y"].to_numpy(float)
                    cases = (time <= horizon) & (event == 1)
                    controls = time > horizon
                    usable = cases | controls
                    ipcw = np.zeros(len(frame), dtype=float)
                    ipcw[cases] = 1.0 / _step_query(*censoring, time[cases], before=True)
                    ipcw[controls] = 1.0 / float(_step_query(*censoring, np.asarray(float(horizon))))
                    risks.append(risk[usable])
                    outcomes.append(cases[usable].astype(float))
                    weights.append(ipcw[usable])
                risk, outcome, ipcw = map(np.concatenate, (risks, outcomes, weights))
                bandwidth = max(1.06 * float(np.std(risk)) * len(risk) ** (-0.2), 0.01)
                for index, value in enumerate(np.quantile(risk, np.linspace(0.05, 0.95, 19)), start=1):
                    kernel = np.exp(-0.5 * ((risk - value) / bandwidth) ** 2) * ipcw
                    rows.append({
                        "model": model, "landmark_years": landmark,
                        "horizon_years": horizon, "point": index,
                        "predicted_risk": float(value),
                        "smoothed_observed_risk": float(np.sum(kernel * outcome) / np.sum(kernel)),
                        "bandwidth": bandwidth,
                    })
    return rows


def run(prediction_root: Path, processed_dir: Path, splits_dir: Path, iterations=2000,
        seed=316, jobs=1, models=None, landmarks=LANDMARKS,
        summary_landmarks=LANDMARKS, cohort="physv11_expanded_ge1y_20260813"):
    models = MODELS if models is None else models
    landmarks = tuple(landmarks)
    summary_landmarks = tuple(summary_landmarks)
    if not set(summary_landmarks).issubset(landmarks):
        raise ValueError("summary landmarks must be a subset of evaluation landmarks")
    cells, patient_ids, prediction_paths = load_data(
        prediction_root, processed_dir, splits_dir, models, landmarks
    )
    _init_worker(cells)
    evaluate = lambda counts: _evaluate_counts(counts, models, landmarks, summary_landmarks)
    observed = evaluate(np.ones(len(patient_ids), dtype=np.int16))
    rng = np.random.default_rng(seed)
    draws = [rng.multinomial(len(patient_ids), np.full(len(patient_ids), 1.0 / len(patient_ids))).astype(np.int16) for _ in range(iterations)]
    if jobs == 1:
        bootstrap = np.asarray([evaluate(draw) for draw in draws])
    else:
        from functools import partial
        evaluate = partial(_evaluate_counts, models=models, landmarks=landmarks,
                           summary_landmarks=summary_landmarks)
        with mp.get_context("fork").Pool(jobs, initializer=_init_worker, initargs=(cells,)) as pool:
            bootstrap = np.asarray(list(pool.imap(evaluate, draws, chunksize=1)))

    levels = [f"landmark_{value}" for value in landmarks] + ["mean"]
    result = {
        "status": "COMPLETE",
        "cohort": cohort,
        "n_patients": len(patient_ids),
        "models": list(models),
        "landmarks": list(landmarks),
        "summary_landmarks": list(summary_landmarks),
        "folds": list(FOLDS),
        "uno_tau_years": 10.0,
        "ibs_interval_years": [1.0, 10.0],
        "ibs_nodes_years": list(IBS_NODES),
        "reported_horizons_years": [5.0, 10.0],
        "internal_guardrail_horizons_years": [3.0],
        "censoring_estimation": "corresponding outer-training landmark risk set",
        "bootstrap": {"unit": "patient", "iterations": iterations, "seed": seed},
        "absolute": {},
        "paired_vs_ddhigan": {},
        "calibration_curves": _calibration_curves(cells, models, landmarks),
        "smoothed_calibration_curves": _smoothed_calibration_curves(cells, models, landmarks),
        "provenance": {
            "prediction_file_count": len(prediction_paths),
            "prediction_set_sha256": hashlib.sha256("".join(f"{path.relative_to(prediction_root).as_posix()} {sha256(path)}\n" for path in sorted(prediction_paths)).encode()).hexdigest(),
            "outcomes_sha256": sha256(processed_dir / "igan_outcomes.csv"),
            "splits_sha256": hashlib.sha256("".join(f"{path.name} {sha256(path)}\n" for path in sorted(splits_dir.glob("fold_*.csv"))).encode()).hexdigest(),
            "probability_roundoff": {
                "tolerance": float(max(
                    cells[(model, fold, landmark)][0].attrs["probability_roundoff"]["tolerance"]
                    for model in models for fold in FOLDS for landmark in landmarks
                )),
                "adjusted_values": int(sum(
                    cells[(model, fold, landmark)][0].attrs["probability_roundoff"]["adjusted_values"]
                    for model in models for fold in FOLDS for landmark in landmarks
                )),
                "adjusted_rows": int(sum(
                    cells[(model, fold, landmark)][0].attrs["probability_roundoff"]["adjusted_rows"]
                    for model in models for fold in FOLDS for landmark in landmarks
                )),
                "maximum_absolute_adjustment": float(max(
                    cells[(model, fold, landmark)][0].attrs["probability_roundoff"]["maximum_absolute_adjustment"]
                    for model in models for fold in FOLDS for landmark in landmarks
                )),
            },
        },
    }
    for model_index, model in enumerate(models):
        result["absolute"][model] = {}
        for level_index, level in enumerate(levels):
            result["absolute"][model][level] = {
                metric: _summary(observed[model_index, level_index, metric_index], bootstrap[:, model_index, level_index, metric_index])
                for metric_index, metric in enumerate(METRICS)
            }
        if model != "DDHIgAN":
            result["paired_vs_ddhigan"][model] = {}
            for level_index, level in enumerate(levels):
                result["paired_vs_ddhigan"][model][level] = {
                    f"delta_{metric}": _summary(
                        observed[model_index, level_index, metric_index] - observed[0, level_index, metric_index],
                        bootstrap[:, model_index, level_index, metric_index] - bootstrap[:, 0, level_index, metric_index],
                    )
                    for metric_index, metric in enumerate(METRICS)
                }
    return canonicalize(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--cohort", default="physv11_expanded_ge1y_20260813")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--landmarks", default="0,1,3,5")
    parser.add_argument("--summary-landmarks", default="0,1,3,5")
    parser.add_argument("--models", help="Comma-separated model names; defaults to the formal seven")
    parser.add_argument("--model", action="append", default=[], help="Custom NAME=DIRECTORY=PREFIX; repeat in display order")
    parser.add_argument("--allhistory-directory")
    parser.add_argument("--allhistory-prefix")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.jobs < 1:
        raise ValueError("iterations and jobs must be positive")
    if args.model and (args.models or args.allhistory_directory or args.allhistory_prefix):
        raise ValueError("custom --model cannot be combined with legacy model selection")
    models = parse_model_specs(args.model) if args.model else select_models(
        args.models, args.allhistory_directory, args.allhistory_prefix
    )
    landmarks = parse_landmarks(args.landmarks)
    summary_landmarks = parse_landmarks(args.summary_landmarks)
    result = run(args.prediction_root, args.processed_dir, args.splits_dir,
                 args.iterations, args.seed, args.jobs, models, landmarks,
                 summary_landmarks, args.cohort)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"UNO_IBS_EVIDENCE_OK models={len(models)} patients={result['n_patients']} iterations={args.iterations}")


if __name__ == "__main__":
    main()
