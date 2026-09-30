import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.metrics import (
    calibration_groups,
    cumulative_dynamic_auc,
    ipcw_calibration_parameters,
    ipcw_brier_scores,
    ipcw_weight_diagnostics,
    uno_c_index,
)
from utils.probabilities import project_probability_roundoff


def make_landmark_outcomes(
    outcomes: pd.DataFrame,
    outcome: str,
    landmark: float,
    max_horizon: float = 10.0,
    absolute_max_time: float = 15.0,
) -> pd.DataFrame:
    time_col = f"{outcome.lower()}_time"
    status_col = f"{outcome.lower()}_status"
    required = {"patient_id", time_col, status_col}
    missing = required.difference(outcomes.columns)
    if missing:
        raise ValueError(f"outcome table is missing columns: {sorted(missing)}")
    result = outcomes.loc[:, list(required)].copy()
    result[time_col] = pd.to_numeric(result[time_col], errors="coerce")
    result[status_col] = pd.to_numeric(result[status_col], errors="coerce")
    result = result.dropna(subset=[time_col, status_col])
    result["absolute_time"] = result[time_col].to_numpy(dtype=float)
    result["event"] = result[status_col].to_numpy(dtype=int)
    result = result[result["absolute_time"] > landmark].copy()
    result["residual_time"] = result["absolute_time"].to_numpy() - landmark
    return result[["patient_id", "residual_time", "event"]]


def _residual_survival_columns(df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    columns = [c for c in df.columns if c.startswith("resid_surv_")]
    columns.sort(key=lambda c: float(c.removeprefix("resid_surv_").removesuffix("y")))
    times = np.array([
        float(c.removeprefix("resid_surv_").removesuffix("y")) for c in columns
    ])
    return columns, times


def evaluate_landmark(
    df: pd.DataFrame,
    train_time: np.ndarray,
    train_event: np.ndarray,
    landmark: float,
    horizons: list[float],
    max_horizon: float,
    calibration_groups_count: int = 5,
) -> tuple[dict, pd.DataFrame]:
    if "residual_time" not in df or "true_event" not in df:
        raise ValueError("prediction data must contain residual_time and true_event")
    df = df.copy()
    probability_columns = [
        column for column in df.columns
        if column == "risk_score" or column.startswith("cond_surv_") or column.startswith("resid_surv_")
    ]
    if not probability_columns:
        raise ValueError("prediction data contains no survival probabilities")
    projected, probability_audit = project_probability_roundoff(
        df[probability_columns].to_numpy(dtype=float)
    )
    df.loc[:, probability_columns] = projected
    test_time = df["residual_time"].to_numpy(dtype=float)
    test_event = df["true_event"].to_numpy(dtype=int)

    available_horizons = [h for h in horizons if f"cond_surv_{int(h)}y" in df]
    if not available_horizons:
        raise ValueError("no requested conditional survival horizons are available")
    overall_horizon = max(available_horizons)
    overall_risk = 1.0 - df[f"cond_surv_{int(overall_horizon)}y"].to_numpy()
    tau = min(float(max_horizon), float(np.max(test_time)))

    metrics = {
        "landmark": float(landmark),
        "n_at_risk": int(len(df)),
        "n_events": int(test_event.sum()),
        "uno_c_index": uno_c_index(
            train_time, train_event, test_time, test_event, overall_risk, tau=tau
        ),
        "uno_tau": tau,
        "overall_risk_horizon": float(overall_horizon),
        "probability_roundoff_tolerance": probability_audit["tolerance"],
        "probability_roundoff_adjusted_values": probability_audit["adjusted_values"],
        "probability_roundoff_adjusted_rows": probability_audit["adjusted_rows"],
        "probability_roundoff_maximum_absolute_adjustment": probability_audit["maximum_absolute_adjustment"],
        "probability_roundoff_pre_projection_minimum": probability_audit["pre_projection_minimum"],
        "probability_roundoff_pre_projection_maximum": probability_audit["pre_projection_maximum"],
    }

    calibration_rows = []
    for horizon in available_horizons:
        risk = 1.0 - df[f"cond_surv_{int(horizon)}y"].to_numpy()
        metrics[f"auc_{int(horizon)}y"] = cumulative_dynamic_auc(
            train_time,
            train_event,
            test_time,
            test_event,
            risk,
            eval_time=horizon,
        )
        intercept, slope = ipcw_calibration_parameters(
            train_time, train_event, test_time, test_event, risk, horizon
        )
        metrics[f"calibration_intercept_{int(horizon)}y"] = intercept
        metrics[f"calibration_slope_{int(horizon)}y"] = slope
        for name, value in ipcw_weight_diagnostics(
            train_time, train_event, test_time, test_event, horizon
        ).items():
            metrics[f"ipcw_weight_{name}_{int(horizon)}y"] = value
        for row in calibration_groups(
            test_time,
            test_event,
            risk,
            eval_time=horizon,
            n_groups=calibration_groups_count,
        ):
            calibration_rows.append({
                "landmark": float(landmark),
                "horizon": float(horizon),
                **row,
            })

    survival_columns, eval_times = _residual_survival_columns(df)
    valid = eval_times <= max_horizon + 1e-9
    survival_columns = [c for c, keep in zip(survival_columns, valid) if keep]
    eval_times = eval_times[valid]
    if len(eval_times) < 2:
        raise ValueError("at least two residual survival grid points are required for IBS")
    brier = ipcw_brier_scores(
        train_time,
        train_event,
        test_time,
        test_event,
        df[survival_columns].to_numpy(dtype=float),
        eval_times,
    )
    integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    metrics["ipcw_ibs"] = float(integrate(brier, eval_times) / (eval_times[-1] - eval_times[0]))
    metrics["ibs_start"] = float(eval_times[0])
    metrics["ibs_end"] = float(eval_times[-1])
    for time, score in zip(eval_times, brier):
        metrics[f"brier_{time:g}y"] = float(score)

    return metrics, pd.DataFrame(calibration_rows)


def load_training_landmark_outcomes(
    processed_dir: str,
    splits_dir: str,
    fold: int | None,
    outcome: str,
    landmark: float,
    max_horizon: float,
    absolute_max_time: float,
    train_partitions: list[int] | None = None,
) -> pd.DataFrame:
    outcomes = pd.read_csv(Path(processed_dir) / "igan_outcomes.csv")
    if train_partitions is not None:
        train_ids = set().union(*(
            set(pd.read_csv(Path(splits_dir) / f"fold_{partition}_test.csv")["patient_id"].astype(str))
            for partition in train_partitions
        ))
    elif fold is not None:
        train_ids = set(
            pd.read_csv(Path(splits_dir) / f"fold_{fold}_train.csv")["patient_id"].astype(str)
        )
    else:
        raise ValueError("either fold or train_partitions is required")
    outcomes = outcomes[outcomes["patient_id"].astype(str).isin(train_ids)]
    return make_landmark_outcomes(
        outcomes,
        outcome,
        landmark,
        max_horizon=max_horizon,
        absolute_max_time=absolute_max_time,
    )


def main():
    parser = argparse.ArgumentParser(description="IPCW landmark survival evaluation.")
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--landmark", type=float, required=True)
    parser.add_argument("--horizons", default="3,5,10")
    parser.add_argument("--max-horizon", type=float, default=10.0)
    parser.add_argument("--absolute-max-time", type=float, default=15.0)
    parser.add_argument("--processed-dir", default="Data/processed")
    parser.add_argument("--splits-dir", default="data/splits")
    split = parser.add_mutually_exclusive_group(required=True)
    split.add_argument("--fold", type=int)
    split.add_argument("--train-partitions", help="comma-separated test-fold partitions used for training")
    parser.add_argument("--outcome", choices=["ESKD", "Drop50"], required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.prediction)
    train_partitions = None
    if args.train_partitions:
        train_partitions = [int(value) for value in args.train_partitions.split(",")]
        if train_partitions != [2, 3, 4]:
            raise ValueError("continuous-time development training partitions must be 2,3,4")
    train = load_training_landmark_outcomes(
        args.processed_dir,
        args.splits_dir,
        args.fold,
        args.outcome,
        args.landmark,
        args.max_horizon,
        args.absolute_max_time,
        train_partitions=train_partitions,
    )
    horizons = [float(value) for value in args.horizons.split(",")]
    metrics, calibration = evaluate_landmark(
        predictions,
        train_time=train["residual_time"].to_numpy(),
        train_event=train["event"].to_numpy(),
        landmark=args.landmark,
        horizons=horizons,
        max_horizon=args.max_horizon,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(metrics, handle, indent=2)
    calibration.to_csv(args.calibration_output, index=False)

    print(
        f"Landmark {args.landmark:g}y: n={metrics['n_at_risk']}, "
        f"Uno C={metrics['uno_c_index']:.3f}, IPCW IBS={metrics['ipcw_ibs']:.3f}"
    )


if __name__ == "__main__":
    main()
