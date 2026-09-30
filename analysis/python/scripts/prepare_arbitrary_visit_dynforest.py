"""Prepare trusted patient-level inputs for the formal DynForest comparator."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from python.scripts.run_arbitrary_visit_reanalysis_references import (
    EVALUATION_LANDMARKS,
    _anchored_longitudinal,
    _base_frame,
)
from python.utils.data_utils import get_fold_split


LONG_COLS = ("CREA", "CystatinC", "ALB", "log_PRO24H")
STATIC_COLS = (
    "age_at_biopsy", "gender", "baseline_CREA", "baseline_CystatinC",
    "baseline_ALB", "baseline_log_PRO24H",
)
FALLBACK = dict(zip(LONG_COLS, STATIC_COLS[2:]))
ARTIFACT_PREFIX = "dynforest_longitudinal_minimal_core_full_outer_train"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _ids_sha256(patient_ids: list[str]) -> str:
    payload = "".join(f"{value}\n" for value in sorted(map(str, patient_ids)))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def full_outer_training_ids(outer_train: list[str], test_ids: list[str], fold: int) -> list[str]:
    fit_ids = list(map(str, outer_train))
    if set(fit_ids).intersection(map(str, test_ids)):
        raise ValueError(f"fold {fold} DynForest fit IDs overlap the test fold")
    return fit_ids


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _time_data(
    anchored: pd.DataFrame, baseline: pd.DataFrame, outcomes: pd.DataFrame,
    patient_ids: list[str], max_time: float, means: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    ids = set(map(str, patient_ids))
    frame = anchored[
        anchored["patient_id"].astype(str).isin(ids)
        & anchored["visit_time"].between(0.0, float(max_time), inclusive="both")
    ][["patient_id", "visit_time", *LONG_COLS]].copy()
    frame["patient_id"] = frame["patient_id"].astype(str)
    outcome = outcomes[["patient_id", "eskd_time"]].copy()
    outcome["patient_id"] = outcome["patient_id"].astype(str)
    frame = frame.merge(outcome, on="patient_id", how="inner", validate="many_to_one")
    frame = frame[frame["visit_time"] < frame["eskd_time"]].drop(columns="eskd_time")
    base = baseline[["patient_id", *FALLBACK.values()]].copy()
    base["patient_id"] = base["patient_id"].astype(str)
    frame = frame.merge(base, on="patient_id", how="left", validate="many_to_one")
    if means is None:
        means = {}
        for column in LONG_COLS:
            values = pd.to_numeric(frame[column], errors="coerce")
            fallback = pd.to_numeric(frame[FALLBACK[column]], errors="coerce")
            combined = pd.concat([values, fallback], ignore_index=True)
            means[column] = float(combined.mean())
    for column in LONG_COLS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        at_zero = frame["visit_time"].eq(0.0) & frame[column].isna()
        frame.loc[at_zero, column] = pd.to_numeric(
            frame.loc[at_zero, FALLBACK[column]], errors="coerce"
        )
        frame.loc[frame["visit_time"].eq(0.0), column] = frame.loc[
            frame["visit_time"].eq(0.0), column
        ].fillna(means[column])
    return frame[["patient_id", "visit_time", *LONG_COLS]], means


def _fixed_data(
    baseline: pd.DataFrame, patient_ids: list[str], means: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    order = pd.DataFrame({"patient_id": list(map(str, patient_ids))})
    base = baseline.copy()
    base["patient_id"] = base["patient_id"].astype(str)
    frame = order.merge(base[["patient_id", *STATIC_COLS]], on="patient_id", how="inner")
    numeric = [column for column in STATIC_COLS if column != "gender"]
    if means is None:
        means = {column: float(pd.to_numeric(frame[column], errors="coerce").mean()) for column in numeric}
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(means[column])
    frame["gender"] = pd.to_numeric(frame["gender"], errors="coerce").fillna(0).astype(int).astype(str)
    return frame, means


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--landmarks", default=",".join(str(int(value)) for value in EVALUATION_LANDMARKS))
    args = parser.parse_args()
    landmarks = tuple(float(value) for value in args.landmarks.split(","))
    if not landmarks or len(set(landmarks)) != len(landmarks):
        raise ValueError("DynForest landmarks must be unique")

    baseline = pd.read_csv(args.processed_dir / "igan_baseline.csv", low_memory=False)
    longitudinal = pd.read_csv(args.processed_dir / "igan_longitudinal.csv", low_memory=False)
    outcomes = pd.read_csv(args.processed_dir / "igan_outcomes.csv", low_memory=False)
    baseline["patient_id"] = baseline["patient_id"].astype(str)
    outcomes["patient_id"] = outcomes["patient_id"].astype(str)
    anchored = _anchored_longitudinal(longitudinal)
    manifest = {
        "status": "READY", "seed": 316, "artifact_prefix": ARTIFACT_PREFIX,
        "evaluation_landmarks": list(landmarks),
        "full_outer_training_required": True, "folds": {},
        "source_input_sha256": {
            name: _sha256(args.processed_dir / name)
            for name in ("igan_baseline.csv", "igan_longitudinal.csv", "igan_outcomes.csv")
        },
        "fold_manifest_sha256": _sha256(args.splits_dir / "fold_manifest.json"),
        "preparation_source_sha256": _sha256(Path(__file__)),
    }

    for fold in map(int, args.folds.split(",")):
        outer_train = get_fold_split(args.splits_dir, fold, "train")
        test_ids = get_fold_split(args.splits_dir, fold, "test")
        train_ids = full_outer_training_ids(outer_train, test_ids, fold)
        fold_dir = args.output_dir / f"fold{fold}"
        train_time, long_means = _time_data(
            anchored, baseline, outcomes, train_ids, 5.0,
        )
        train_fixed, static_means = _fixed_data(baseline, train_ids)
        train_y = _base_frame(baseline, outcomes, train_ids, "ESKD")[[
            "patient_id", "eskd_time", "eskd_status"
        ]]
        for frame_name, frame in (("fixed", train_fixed), ("outcome", train_y)):
            frame_ids = frame["patient_id"].astype(str).tolist()
            if set(frame_ids) != set(train_ids) or len(frame_ids) != len(train_ids):
                raise ValueError(
                    f"fold {fold} DynForest {frame_name} fit IDs differ from outer-training IDs"
                )
        longitudinal_ids = set(train_time["patient_id"].astype(str))
        if longitudinal_ids != set(train_ids):
            raise ValueError(
                f"fold {fold} DynForest longitudinal fit IDs differ from outer-training IDs"
            )
        _write_csv(train_time, fold_dir / "train_time.csv")
        _write_csv(train_fixed, fold_dir / "train_fixed.csv")
        _write_csv(train_y, fold_dir / "train_outcome.csv")
        test_base = _base_frame(baseline, outcomes, test_ids, "ESKD")
        for landmark in landmarks:
            truth = test_base[test_base["eskd_time"] > landmark].copy()
            ids = truth["patient_id"].astype(str).tolist()
            test_time, _ = _time_data(
                anchored, baseline, outcomes, ids, landmark, long_means,
            )
            test_fixed, _ = _fixed_data(baseline, ids, static_means)
            truth = pd.DataFrame({
                "patient_id": truth["patient_id"].astype(str),
                "true_time": truth["eskd_time"].astype(float),
                "residual_time": truth["eskd_time"].astype(float) - landmark,
                "true_event": truth["eskd_status"].astype(int),
                "Tstart": float(landmark),
            })
            target = fold_dir / f"landmark{int(landmark)}"
            _write_csv(test_time, target / "test_time.csv")
            _write_csv(test_fixed, target / "test_fixed.csv")
            _write_csv(truth, target / "truth.csv")
        (fold_dir / "imputation.json").write_text(json.dumps({
            "longitudinal_t0_means": long_means,
            "static_means": static_means,
        }, indent=2) + "\n")
        prepared_paths = sorted([
            *fold_dir.rglob("*.csv"), fold_dir / "imputation.json",
        ])
        manifest["folds"][str(fold)] = {
            "outer_train_patients": len(outer_train),
            "fit_patients": len(train_ids),
            "excluded_outer_train_patients": 0,
            "validation_patients": 0,
            "train_longitudinal_rows": len(train_time),
            "test_patients": len(test_ids),
            "fit_test_overlap": 0,
            "outer_train_ids_sha256": _ids_sha256(outer_train),
            "fit_ids_sha256": _ids_sha256(train_ids),
            "test_ids_sha256": _ids_sha256(test_ids),
            "split_train_sha256": _sha256(args.splits_dir / f"fold_{fold}_train.csv"),
            "split_test_sha256": _sha256(args.splits_dir / f"fold_{fold}_test.csv"),
            "prepared_input_sha256": {
                path.relative_to(args.output_dir).as_posix(): _sha256(path)
                for path in prepared_paths
            },
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
