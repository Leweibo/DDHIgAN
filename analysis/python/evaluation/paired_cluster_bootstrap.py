from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import calibration_groups

METRICS = (
    "uno_c_index", "ipcw_ibs", "auc_3y", "auc_5y", "auc_10y",
    "calibration_mae_3y", "calibration_mae_5y", "calibration_mae_10y",
)
PRIMARY_AUC10_THRESHOLD = 0.010

_BOOTSTRAP_CANDIDATE = None
_BOOTSTRAP_REFERENCE = None
_BOOTSTRAP_LANDMARKS = None
_BOOTSTRAP_HORIZONS = None


def canonicalize(value):
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {key: canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def _validate_arrays(time, event):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if time.ndim != 1 or event.ndim != 1 or len(time) != len(event):
        raise ValueError("survival arrays must be one-dimensional with equal length")
    if not np.isfinite(time).all() or (time < 0).any():
        raise ValueError("survival time must be finite and non-negative")
    if not np.isin(event, [0, 1]).all():
        raise ValueError("event must contain only zero and one")
    return time, event


def _censoring_step(time, event):
    time, event = _validate_arrays(time, event)
    order = np.argsort(time, kind="stable")
    sorted_time = time[order]
    sorted_censor = 1 - event[order]
    unique_time, starts = np.unique(sorted_time, return_index=True)
    censored = np.add.reduceat(sorted_censor, starts)
    at_risk = len(time) - starts
    survival = np.cumprod(1.0 - censored / at_risk)
    return unique_time, survival


def _step_query(times, survival, query, before=False):
    query = np.asarray(query, dtype=float)
    side = "left" if before else "right"
    indices = np.searchsorted(times, query, side=side) - 1
    result = np.ones(query.shape, dtype=float)
    valid = indices >= 0
    result[valid] = survival[indices[valid]]
    if np.any(result <= 1e-12):
        raise ValueError("censoring survival is zero; evaluation is unsupported")
    return result


def _fast_uno_c_index(time, event, risk, tau):
    time, event = _validate_arrays(time, event)
    risk = np.asarray(risk, dtype=float)
    censor_time, censor_survival = _censoring_step(time, event)
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
    concordant = 0.0
    comparable = 0.0
    start = 0
    while start < len(order):
        stop = start + 1
        group_time = time[order[start]]
        while stop < len(order) and time[order[stop]] == group_time:
            stop += 1
        later_count = query(len(unique_risk) - 1)
        if group_time <= tau and later_count:
            group = order[start:stop]
            event_group = group[event[group] == 1]
            if len(event_group):
                g = _step_query(
                    censor_time, censor_survival, time[event_group], before=True
                )
                for index, weight in zip(event_group, 1.0 / (g * g)):
                    rank = ranks[index]
                    lower = query(rank - 1)
                    equal = query(rank) - lower
                    concordant += float(weight) * (lower + 0.5 * equal)
                    comparable += float(weight) * later_count
        for index in order[start:stop]:
            update(ranks[index])
        start = stop
    if comparable == 0:
        raise ValueError("no comparable pairs for Uno C-index")
    return concordant / comparable


def _fast_auc(time, event, risk, eval_time):
    time, event = _validate_arrays(time, event)
    risk = np.asarray(risk, dtype=float)
    cases = np.flatnonzero((time <= eval_time) & (event == 1))
    controls = np.flatnonzero(time > eval_time)
    if len(cases) == 0 or len(controls) == 0:
        raise ValueError("dynamic AUC requires at least one case and one control")
    censor_time, censor_survival = _censoring_step(time, event)
    weights = 1.0 / _step_query(
        censor_time, censor_survival, time[cases], before=True
    )
    control_risk = np.sort(risk[controls])
    lower = np.searchsorted(control_risk, risk[cases], side="left")
    upper = np.searchsorted(control_risk, risk[cases], side="right")
    concordance = lower + 0.5 * (upper - lower)
    return float(np.sum(weights * concordance) / (np.sum(weights) * len(controls)))


def _fast_ibs(time, event, predicted_survival, eval_times):
    time, event = _validate_arrays(time, event)
    predicted_survival = np.asarray(predicted_survival, dtype=float)
    eval_times = np.asarray(eval_times, dtype=float)
    censor_time, censor_survival = _censoring_step(time, event)
    scores = []
    for column, eval_time in enumerate(eval_times):
        event_before = (time <= eval_time) & (event == 1)
        event_free = time > eval_time
        weights = np.zeros(len(time), dtype=float)
        if event_before.any():
            weights[event_before] = 1.0 / _step_query(
                censor_time, censor_survival, time[event_before], before=True
            )
        if event_free.any():
            weights[event_free] = 1.0 / float(
                _step_query(censor_time, censor_survival, np.asarray(eval_time))
            )
        observed_survival = event_free.astype(float)
        scores.append(np.mean(
            weights * (observed_survival - predicted_survival[:, column]) ** 2
        ))
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(integrate(scores, eval_times) / (eval_times[-1] - eval_times[0]))


def cluster_bootstrap_patient_ids(
    patient_ids: list[str], iterations: int = 2000, seed: int = 316
) -> list[np.ndarray]:
    unique = np.asarray(sorted(set(map(str, patient_ids))), dtype=object)
    if len(unique) < 2:
        raise ValueError("at least two unique patients are required")
    rng = np.random.default_rng(seed)
    return [rng.choice(unique, size=len(unique), replace=True) for _ in range(iterations)]


def _resample_rows(frame: pd.DataFrame, sampled_ids: np.ndarray) -> pd.DataFrame:
    patient_ids = frame["patient_id"].astype(str).to_numpy()
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("landmark frame contains duplicate patient_id")
    row_by_patient = {patient_id: index for index, patient_id in enumerate(patient_ids)}
    selected = [
        (draw, str(patient_id), row_by_patient[str(patient_id)])
        for draw, patient_id in enumerate(sampled_ids)
        if str(patient_id) in row_by_patient
    ]
    if not selected:
        raise ValueError("bootstrap draw contains no patients at risk")
    result = frame.iloc[[item[2] for item in selected]].reset_index(drop=True).copy()
    result["bootstrap_patient_id"] = [
        f"{patient_id}__draw{draw}" for draw, patient_id, _ in selected
    ]
    return result


def _evaluate(frame: pd.DataFrame, landmark: float, horizons: list[float]) -> dict:
    # The resampled OOF cohort supplies the censoring distribution anew in every draw.
    time = frame["residual_time"].to_numpy(dtype=float)
    event = frame["true_event"].to_numpy(dtype=int)
    overall_risk = 1.0 - frame[f"cond_surv_{int(max(horizons))}y"].to_numpy(dtype=float)
    tau = min(float(max(horizons)), float(np.max(time)))
    metrics = {
        "uno_c_index": _fast_uno_c_index(time, event, overall_risk, tau),
    }
    calibration_rows = []
    for horizon in horizons:
        risk = 1.0 - frame[f"cond_surv_{int(horizon)}y"].to_numpy(dtype=float)
        metrics[f"auc_{int(horizon)}y"] = _fast_auc(time, event, risk, horizon)
        for row in calibration_groups(time, event, risk, horizon, n_groups=5):
            calibration_rows.append({"horizon": horizon, **row})
    survival_columns = [column for column in frame if column.startswith("resid_surv_")]
    survival_columns.sort(
        key=lambda column: float(column.removeprefix("resid_surv_").removesuffix("y"))
    )
    eval_times = np.asarray([
        float(column.removeprefix("resid_surv_").removesuffix("y"))
        for column in survival_columns
    ])
    valid = eval_times <= max(horizons) + 1e-9
    survival_columns = [column for column, keep in zip(survival_columns, valid) if keep]
    eval_times = eval_times[valid]
    metrics["ipcw_ibs"] = _fast_ibs(
        time, event, frame[survival_columns].to_numpy(dtype=float), eval_times
    )
    calibration = pd.DataFrame(calibration_rows)
    for horizon in horizons:
        rows = calibration[calibration["horizon"].astype(float) == horizon]
        metrics[f"calibration_mae_{int(horizon)}y"] = float(
            np.mean(np.abs(rows["mean_predicted_risk"] - rows["observed_risk"]))
        )
    return metrics


def _init_bootstrap_worker(candidate, reference, landmarks, horizons):
    global _BOOTSTRAP_CANDIDATE, _BOOTSTRAP_REFERENCE
    global _BOOTSTRAP_LANDMARKS, _BOOTSTRAP_HORIZONS
    _BOOTSTRAP_CANDIDATE = candidate
    _BOOTSTRAP_REFERENCE = reference
    _BOOTSTRAP_LANDMARKS = landmarks
    _BOOTSTRAP_HORIZONS = horizons


def _bootstrap_draw(sampled_ids):
    result = {}
    per_landmark = {metric: [] for metric in METRICS}
    for landmark in _BOOTSTRAP_LANDMARKS:
        cand = _resample_rows(
            _BOOTSTRAP_CANDIDATE[
                _BOOTSTRAP_CANDIDATE["Tstart"].astype(float) == landmark
            ],
            sampled_ids,
        )
        ref = _resample_rows(
            _BOOTSTRAP_REFERENCE[
                _BOOTSTRAP_REFERENCE["Tstart"].astype(float) == landmark
            ],
            sampled_ids,
        )
        cand_metrics = _evaluate(cand, landmark, _BOOTSTRAP_HORIZONS)
        ref_metrics = _evaluate(ref, landmark, _BOOTSTRAP_HORIZONS)
        for metric in METRICS:
            delta = float(cand_metrics[metric] - ref_metrics[metric])
            per_landmark[metric].append(delta)
            result[f"landmark_{landmark:g}_delta_{metric}"] = delta
    for metric in METRICS:
        result[f"mean_delta_{metric}"] = float(np.mean(per_landmark[metric]))
    return result


def paired_cluster_bootstrap(
    candidate: pd.DataFrame,
    reference: pd.DataFrame,
    iterations: int = 2000,
    seed: int = 316,
    horizons: list[float] | None = None,
    jobs: int = 1,
) -> dict:
    horizons = horizons or [3.0, 5.0, 10.0]
    if jobs < 1:
        raise ValueError("jobs must be at least one")
    keys = ["patient_id", "Tstart"]
    if candidate.duplicated(keys).any() or reference.duplicated(keys).any():
        raise ValueError("each model must contain one row per patient/landmark")
    candidate_keys = set(map(tuple, candidate[keys].astype({"patient_id": str}).to_numpy()))
    reference_keys = set(map(tuple, reference[keys].astype({"patient_id": str}).to_numpy()))
    if candidate_keys != reference_keys:
        raise ValueError("candidate/reference patient-landmark sets differ")
    draws = cluster_bootstrap_patient_ids(
        candidate["patient_id"].astype(str).tolist(), iterations, seed
    )
    landmarks = sorted(candidate["Tstart"].astype(float).unique())
    distributions = {
        f"mean_delta_{metric}": [] for metric in METRICS
    }
    landmark_distributions = {
        f"landmark_{landmark:g}_delta_{metric}": []
        for landmark in landmarks for metric in METRICS
    }
    observed = {name: None for name in distributions}
    observed.update({name: None for name in landmark_distributions})
    observed_by_metric = {metric: [] for metric in METRICS}
    for landmark in landmarks:
        cand_metrics = _evaluate(
            candidate[candidate["Tstart"].astype(float) == landmark], landmark, horizons
        )
        ref_metrics = _evaluate(
            reference[reference["Tstart"].astype(float) == landmark], landmark, horizons
        )
        for metric in METRICS:
            delta = float(cand_metrics[metric] - ref_metrics[metric])
            observed[f"landmark_{landmark:g}_delta_{metric}"] = delta
            observed_by_metric[metric].append(delta)
    for metric in METRICS:
        observed[f"mean_delta_{metric}"] = float(np.mean(observed_by_metric[metric]))
    _init_bootstrap_worker(candidate, reference, landmarks, horizons)
    if jobs == 1:
        bootstrap_results = map(_bootstrap_draw, draws)
        pool = None
    else:
        context = mp.get_context("fork")
        pool = context.Pool(
            processes=jobs,
            initializer=_init_bootstrap_worker,
            initargs=(candidate, reference, landmarks, horizons),
        )
        bootstrap_results = pool.imap(_bootstrap_draw, draws, chunksize=1)
    try:
        for draw_result in bootstrap_results:
            for name in distributions:
                distributions[name].append(draw_result[name])
            for name in landmark_distributions:
                landmark_distributions[name].append(draw_result[name])
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    distributions.update(landmark_distributions)
    summary = {}
    for name, values in distributions.items():
        array = np.asarray(values, dtype=float)
        summary[name] = {
            "mean": float(observed[name]),
            "bootstrap_mean": float(array.mean()),
            "ci_lower": float(np.quantile(array, 0.025)),
            "ci_upper": float(np.quantile(array, 0.975)),
        }
    return {"seed": seed, "iterations": iterations, "jobs": jobs, "summary": summary}


def advancement_decision(bootstrap: dict) -> dict:
    summary = bootstrap["summary"]
    auc10 = summary["mean_delta_auc_10y"]
    landmark_auc10 = [
        value for key, value in summary.items()
        if key.startswith("landmark_") and key.endswith("delta_auc_10y")
    ]
    landmark_uno = [
        value for key, value in summary.items()
        if key.startswith("landmark_") and key.endswith("delta_uno_c_index")
    ]
    landmark_ibs = [
        value for key, value in summary.items()
        if key.startswith("landmark_") and key.endswith("delta_ipcw_ibs")
    ]
    direction_count = sum(item["mean"] > 0 for item in landmark_auc10)
    primary_pass = (
        auc10["mean"] >= PRIMARY_AUC10_THRESHOLD
        and auc10["ci_lower"] > 0
        and direction_count >= 3
    )
    no_harm = all(item["mean"] >= -0.010 for item in landmark_uno) and all(
        item["mean"] <= 0.002 for item in landmark_ibs
    )
    auc_short_no_harm = (
        summary["mean_delta_auc_3y"]["mean"] >= -0.005
        and summary["mean_delta_auc_5y"]["mean"] >= -0.005
    )
    calibration_no_harm = all(
        summary[f"mean_delta_calibration_mae_{horizon}y"]["mean"] <= 0.005
        for horizon in (3, 5, 10)
    )
    passed = bool(primary_pass and no_harm and auc_short_no_harm and calibration_no_harm)
    return {
        "advance": passed,
        "primary_metric": "mean_paired_delta_auc_10y",
        "primary_threshold": PRIMARY_AUC10_THRESHOLD,
        "primary_threshold_display": f"{PRIMARY_AUC10_THRESHOLD:.3f}",
        "primary_threshold_passed": primary_pass,
        "auc10_positive_landmarks": direction_count,
        "landmark_no_harm": no_harm,
        "auc3_auc5_no_harm": auc_short_no_harm,
        "calibration_no_harm": calibration_no_harm,
        "interpretation": (
            "candidate model advances" if passed
            else "retain minimal-core clinical Dynamic-DeepHit as the primary model"
        ),
    }


def _load_oof(directory: str, prefix: str, outcome: str) -> pd.DataFrame:
    frames = []
    for fold in range(5):
        for landmark in (0, 1, 3, 5):
            path = Path(directory) / f"{prefix}_{outcome}_fold{fold}_test_landmark{landmark}.csv"
            frame = pd.read_csv(path, dtype={"patient_id": str})
            frame["fold"] = fold
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Patient-cluster bootstrap of OOF predictions")
    parser.add_argument("--candidate-dir", required=True)
    parser.add_argument("--candidate-prefix", required=True)
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--reference-prefix", required=True)
    parser.add_argument("--outcome", default="ESKD")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=316)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--skip-advancement-decision", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    candidate = _load_oof(args.candidate_dir, args.candidate_prefix, args.outcome)
    reference = _load_oof(args.reference_dir, args.reference_prefix, args.outcome)
    result = paired_cluster_bootstrap(
        candidate, reference, args.iterations, args.seed, jobs=args.jobs
    )
    if not args.skip_advancement_decision:
        result["advancement_decision"] = advancement_decision(result)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(canonicalize(result), handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
