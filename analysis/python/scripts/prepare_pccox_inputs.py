"""Prepare fold-isolated, actual-visit PCCox inputs without fitting a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from python.scripts.run_arbitrary_visit_reanalysis_references import (
    CURVE_TIMES,
    EVALUATION_LANDMARKS,
    build_actual_visit_training_frame,
    build_landmark_evaluation_frame,
    build_pccox_actual_visit_training_frame,
    impute_from_training,
)
from python.utils.data_utils import get_fold_split

FEATURES = ["age_at_biopsy", "gender", "CREA", "CystatinC", "ALB", "log_PRO24H"]


def prepare_fold(
    processed_dir: Path, splits_dir: Path, output_dir: Path, fold: int,
    landmarks: tuple[float, ...] = EVALUATION_LANDMARKS,
) -> dict:
    baseline = pd.read_csv(processed_dir / "igan_baseline.csv", low_memory=False)
    longitudinal = pd.read_csv(processed_dir / "igan_longitudinal.csv", low_memory=False)
    outcomes = pd.read_csv(processed_dir / "igan_outcomes.csv", dtype={"patient_id": str})
    outer_train = get_fold_split(splits_dir, fold, "train")
    test_ids = get_fold_split(splits_dir, fold, "test")
    raw_fit = build_pccox_actual_visit_training_frame(
        baseline, longitudinal, outcomes, outer_train, "ESKD", 5.0,
    )
    fit, _ = impute_from_training(raw_fit, raw_fit, FEATURES)
    imputation = {column: float(fit[column].mean()) for column in FEATURES}
    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=False)
    fit.to_csv(fold_dir / "fit.csv", index=False)

    cells = []
    for landmark in landmarks:
        test = build_landmark_evaluation_frame(
            baseline, longitudinal, outcomes, test_ids, landmark, "ESKD",
        )
        _, test = impute_from_training(fit, test, FEATURES)
        test.rename(columns={"query_time": "measurement_time"}).to_csv(
            fold_dir / f"test_landmark{int(landmark)}.csv", index=False,
        )
        cells.append({"landmark": landmark, "n": len(test)})
    if not (fit["measurement_time"].between(0, 5).all() and (fit["measurement_time"] < fit["stime"]).all()):
        raise ValueError(f"fold {fold}: invalid PCCox training query times")
    if fit.duplicated(["patient_id", "measurement_time"]).any():
        raise ValueError(f"fold {fold}: duplicate patient measurement time")
    metadata = {
        "fold": fold, "seed": 316, "fit_count": 1,
        "fit_patients": int(fit.patient_id.nunique()), "fit_rows": len(fit),
        "test_patients": len(test_ids), "landmarks": cells,
        "training_query_window_years": [0.0, 5.0],
        "prediction_times_years": list(CURVE_TIMES),
        "features": FEATURES, "imputation": imputation,
        "uses_patient_visit_weights": False, "uses_penalizer": False,
        "uses_blup": False, "measurement_time_spline_df": 3,
    }
    (fold_dir / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--landmarks", default=",".join(str(int(value)) for value in EVALUATION_LANDMARKS))
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing to overwrite PCCox prepared inputs")
    args.output_dir.mkdir(parents=True)
    landmarks = tuple(float(value) for value in args.landmarks.split(","))
    if not landmarks or len(set(landmarks)) != len(landmarks):
        raise ValueError("PCCox landmarks must be unique")
    rows = [prepare_fold(args.processed_dir, args.splits_dir, args.output_dir, int(fold), landmarks)
            for fold in args.folds.split(",")]
    (args.output_dir / "manifest.json").write_text(json.dumps({"folds": rows}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
