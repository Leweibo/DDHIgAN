"""Date-free recent-follow-up sequences for the exploratory DDHIgAN branch.

The private source tables contain dates solely to calculate chronological age,
the query-relative outcome, and the trailing 3- or 5-year window.  This module
never places dates, biopsy timing, or a first-visit offset in the model-facing
sequence table.  It is deliberately separate from the biopsy-anchored and
first-visit/all-history pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:  # Snapshot construction is intentionally usable without a training runtime.
    import torch
    from torch.utils.data import Dataset
except ModuleNotFoundError:  # pragma: no cover - exercised on date-build hosts
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass

from .firstvisit_allhistory import CORE_MARKERS, fixed_span_visit_clusters
from python.utils.time_grid import HalfYearTimeGrid


DAYS_PER_YEAR = 365.0
WINDOWS = (3, 5)
MAX_VISITS = 64
STATIC_COLUMNS = ("age_at_query", "gender")
SEQUENCE_COLUMNS = (
    "patient_id", "query_id", "sequence_index", "age_at_query", "gender",
    "intervisit_gap_years", *CORE_MARKERS, "residual_time", "event_status",
)
PRIVATE_DATE_COLUMNS = (
    "patient_id", "query_id", "query_date", "endpoint_date",
    "residual_time", "event_status",
)


class RecentFollowupOverflowError(ValueError):
    """Raised before writing a snapshot when a trailing window exceeds 64 visits."""

    def __init__(self, preflight: dict):
        self.preflight = preflight
        super().__init__(
            "recent-follow-up window exceeds the 64-visit fail-closed limit; "
            "no model snapshot may be written"
        )


def _normal_date(frame: pd.DataFrame, column: str, name: str) -> pd.Series:
    value = pd.to_datetime(frame[column], errors="coerce").dt.normalize()
    if value.isna().any():
        raise ValueError(f"{name} must contain finite calendar dates in private staging")
    return value


def _validate_private_sources(
    baseline: pd.DataFrame,
    longitudinal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    expected_patients: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_baseline = {"patient_id", "birth_date", "gender"}
    required_longitudinal = {"patient_id", "visit_date", *CORE_MARKERS}
    required_outcomes = {"patient_id", "endpoint_date", "eskd_status"}
    for name, frame, required in (
        ("baseline", baseline, required_baseline),
        ("longitudinal", longitudinal, required_longitudinal),
        ("outcomes", outcomes, required_outcomes),
    ):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"private {name} source is missing columns: {sorted(missing)}")

    base = baseline.copy()
    long = longitudinal.copy()
    out = outcomes.copy()
    for frame in (base, long, out):
        frame["patient_id"] = frame["patient_id"].astype(str)
        if frame["patient_id"].eq("").any():
            raise ValueError("private source contains an empty patient_id")
    if base["patient_id"].duplicated().any() or out["patient_id"].duplicated().any():
        raise ValueError("private baseline/outcome source has duplicate patient_id")
    if len(base) != expected_patients or set(base["patient_id"]) != set(out["patient_id"]):
        raise ValueError(
            f"private 9,948-patient key gate failed: baseline={len(base)} outcomes={len(out)}"
        )

    base["birth_date"] = _normal_date(base, "birth_date", "birth_date")
    long["visit_date"] = _normal_date(long, "visit_date", "visit_date")
    out["endpoint_date"] = _normal_date(out, "endpoint_date", "endpoint_date")
    base["gender"] = pd.to_numeric(base["gender"], errors="coerce")
    out["eskd_status"] = pd.to_numeric(out["eskd_status"], errors="coerce")
    if not np.isfinite(base["gender"].to_numpy(float)).all() or not base["gender"].isin([0, 1]).all():
        raise ValueError("gender must be binary and finite")
    if not out["eskd_status"].isin([0, 1]).all():
        raise ValueError("rechecked ESKD status must be binary")
    for marker in CORE_MARKERS:
        value = pd.to_numeric(long[marker], errors="coerce")
        if np.isinf(value).any():
            raise ValueError(f"{marker} contains an infinite value")
        long[marker] = value
    if not set(long["patient_id"]).issubset(set(base["patient_id"])):
        raise ValueError("longitudinal source contains a patient outside the frozen cohort")
    return base, long, out


def _valid_query_mask(clustered: pd.DataFrame) -> pd.Series:
    return clustered["CREA"].notna() & clustered[["CystatinC", "ALB", "log_PRO24H"]].notna().any(axis=1)


def _cluster_with_private_context(
    baseline: pd.DataFrame,
    longitudinal: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    clustered = fixed_span_visit_clusters(longitudinal)
    context = baseline[["patient_id", "birth_date", "gender"]].merge(
        outcomes[["patient_id", "endpoint_date", "eskd_status"]],
        on="patient_id", how="inner", validate="one_to_one",
    )
    clustered = clustered.merge(context, on="patient_id", how="inner", validate="many_to_one")
    # The cluster-end date is the query date.  A cluster ending at/after the
    # event/censor date is excluded even if an earlier raw row was in it.
    clustered = clustered[clustered["visit_date"] < clustered["endpoint_date"]].copy()
    clustered["is_valid_query"] = _valid_query_mask(clustered)
    return clustered.sort_values(["patient_id", "visit_date"], kind="stable").reset_index(drop=True)


def preflight_recent_followup_windows(
    baseline: pd.DataFrame,
    longitudinal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    expected_patients: int = 9948,
    windows: tuple[int, ...] = WINDOWS,
    max_visits: int = MAX_VISITS,
) -> tuple[pd.DataFrame, dict]:
    """Return private clusters and only aggregate window-limit information.

    This preflight is intentionally completed for *both* requested windows
    before either snapshot can be written.  It reports counts only, permitting
    an overflow audit without silently retaining the most recent 64 visits.
    """
    windows = tuple(int(value) for value in windows)
    if set(windows) != set(WINDOWS):
        raise ValueError("recent-follow-up preflight is locked to the 3- and 5-year windows")
    if max_visits != MAX_VISITS:
        raise ValueError("recent-follow-up max_visits is locked to 64")
    base, long, out = _validate_private_sources(
        baseline, longitudinal, outcomes, expected_patients=expected_patients,
    )
    clustered = _cluster_with_private_context(base, long, out)
    summary = {}
    for window in sorted(windows):
        rows = []
        for _, group in clustered.groupby("patient_id", sort=False):
            valid = group.loc[group["is_valid_query"]]
            for query in valid["visit_date"]:
                count = int(((group["visit_date"] >= query - pd.Timedelta(days=window * DAYS_PER_YEAR))
                             & (group["visit_date"] <= query)).sum())
                rows.append((str(group["patient_id"].iloc[0]), count))
        counts = np.asarray([count for _, count in rows], dtype=int)
        summary[str(window)] = {
            "window_years": window,
            "eligible_queries": int(len(rows)),
            "patients_with_eligible_queries": int(len({patient for patient, _ in rows})),
            "max_visits_in_window": int(counts.max()) if len(counts) else 0,
            "queries_over_64": int((counts > max_visits).sum()),
            "patients_over_64": int(len({patient for patient, count in rows if count > max_visits})),
            "truncation": "prohibited",
        }
    preflight = {
        "status": "READY" if all(value["queries_over_64"] == 0 for value in summary.values()) else "VISIT_LIMIT_EXCEEDED",
        "patients": expected_patients,
        "cluster_contract": "non-transitive fixed 15-day span; timestamp=cluster end",
        "query_contract": "CREA plus >=1 of CystatinC/ALB/log_PRO24H before rechecked ESKD/censor date",
        "windows": summary,
        "max_visits": max_visits,
    }
    return clustered, preflight


@dataclass(frozen=True)
class RecentFollowupSnapshot:
    sequences: pd.DataFrame
    patient_outcomes: pd.DataFrame
    evaluation_strata: pd.DataFrame
    private_date_validation: pd.DataFrame
    audit: dict


def build_recent_followup_snapshot(
    baseline: pd.DataFrame,
    longitudinal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    window_years: int,
    expected_patients: int = 9948,
    max_visits: int = MAX_VISITS,
    version: str = "ddhigan_recent_followup_unversioned",
) -> RecentFollowupSnapshot:
    """Build one date-free query-sequence snapshot after the dual preflight.

    Every valid cluster is a query anchor.  The rows retained for that query
    are the clusters in the requested trailing window, and the first encoded
    interval is exactly zero.  Biopsy information is neither required nor
    accepted by this Python builder; the private R staging layer has already
    converted rechecked outcomes to their endpoint calendar dates.
    """
    if int(window_years) not in WINDOWS:
        raise ValueError("window_years must be 3 or 5")
    clustered, preflight = preflight_recent_followup_windows(
        baseline, longitudinal, outcomes,
        expected_patients=expected_patients, max_visits=max_visits,
    )
    if preflight["status"] != "READY":
        raise RecentFollowupOverflowError(preflight)

    sequence_rows: list[dict] = []
    private_rows: list[dict] = []
    strata_rows: list[dict] = []
    query_id = 0
    patients_with_queries: set[str] = set()
    for patient_id, group in clustered.groupby("patient_id", sort=True):
        group = group.sort_values("visit_date", kind="stable").reset_index(drop=True)
        valid = group.loc[group["is_valid_query"]]
        first_valid_date = valid["visit_date"].iloc[0] if len(valid) else None
        for query in valid.itertuples(index=False):
            retained = group.loc[
                (group["visit_date"] >= query.visit_date - pd.Timedelta(days=int(window_years) * DAYS_PER_YEAR))
                & (group["visit_date"] <= query.visit_date)
            ].copy()
            if retained.empty or retained["visit_date"].iloc[-1] != query.visit_date:
                raise ValueError("query cluster must be the final retained window visit")
            if len(retained) > max_visits:
                raise RecentFollowupOverflowError(preflight)
            gaps = np.zeros(len(retained), dtype=float)
            if len(retained) > 1:
                gaps[1:] = np.diff(retained["visit_date"].to_numpy(dtype="datetime64[D]")).astype("timedelta64[D]").astype(float) / DAYS_PER_YEAR
            if not np.isfinite(gaps).all() or np.any(gaps < 0) or gaps[0] != 0.0:
                raise ValueError("recent sequence gap contract failed")
            residual = (query.endpoint_date - query.visit_date).days / DAYS_PER_YEAR
            age = (query.visit_date - query.birth_date).days / DAYS_PER_YEAR
            if not np.isfinite(residual) or residual <= 0 or not np.isfinite(age) or age < 0:
                raise ValueError("private age/outcome date reconstruction failed")
            for sequence_index, (_, row) in enumerate(retained.iterrows()):
                sequence_rows.append({
                    "patient_id": str(patient_id), "query_id": query_id,
                    "sequence_index": sequence_index, "age_at_query": age,
                    "gender": float(query.gender), "intervisit_gap_years": float(gaps[sequence_index]),
                    **{marker: row[marker] for marker in CORE_MARKERS},
                    "residual_time": residual, "event_status": int(query.eskd_status),
                })
            elapsed = (query.visit_date - first_valid_date).days / DAYS_PER_YEAR
            strata_rows.append({
                "patient_id": str(patient_id), "query_id": query_id,
                "first_valid_followup_elapsed_years": elapsed,
                "stratum": "<1" if elapsed < 1 else "1-<3" if elapsed < 3 else ">=3",
            })
            private_rows.append({
                "patient_id": str(patient_id), "query_id": query_id,
                "query_date": query.visit_date, "endpoint_date": query.endpoint_date,
                "residual_time": residual, "event_status": int(query.eskd_status),
            })
            patients_with_queries.add(str(patient_id))
            query_id += 1

    sequences = pd.DataFrame(sequence_rows, columns=SEQUENCE_COLUMNS)
    private_dates = pd.DataFrame(private_rows, columns=PRIVATE_DATE_COLUMNS)
    strata = pd.DataFrame(strata_rows)
    if sequences.empty:
        raise ValueError("no eligible recent-follow-up query visits were constructed")
    _validate_model_sequence_table(sequences, int(window_years), max_visits)
    reconstructed = (
        pd.to_datetime(private_dates["endpoint_date"]).sub(pd.to_datetime(private_dates["query_date"])).dt.days
        / DAYS_PER_YEAR
    )
    if not np.allclose(reconstructed, private_dates["residual_time"].to_numpy(float), atol=0.0, rtol=0.0):
        raise ValueError("residual outcome time does not equal the private calendar-date difference")
    patient_outcomes = pd.DataFrame({
        "patient_id": sorted(set(baseline["patient_id"].astype(str))),
    }).merge(
        outcomes[["patient_id", "eskd_status"]].assign(patient_id=lambda x: x["patient_id"].astype(str)),
        on="patient_id", how="left", validate="one_to_one",
    ).sort_values("patient_id", kind="stable")
    audit = {
        "status": "READY_FOR_HUMAN_CONFIRMATION",
        "version": version,
        "patients": expected_patients,
        "patients_with_queries": len(patients_with_queries),
        "eligible_queries": int(query_id),
        "sequence_rows": int(len(sequences)),
        "window_years": int(window_years),
        "max_visits": max_visits,
        "preflight": preflight,
        "static_inputs": list(STATIC_COLUMNS),
        "longitudinal_inputs": list(CORE_MARKERS),
        "time_encoding": "inter-cluster gaps only; first gap=0; no absolute or biopsy-relative time",
        "outcome_contract": "rechecked kidney failure (ESKD) residual time from private endpoint calendar date",
        "date_fields_in_model_sequence_table": False,
        "biopsy_fields_in_model_sequence_table": False,
        "private_date_validation_excluded_from_model": True,
        "evaluation_strata_not_model_input": True,
        "training_started": False,
    }
    return RecentFollowupSnapshot(sequences, patient_outcomes, strata, private_dates, audit)


def _validate_model_sequence_table(
    sequences: pd.DataFrame,
    window_years: int,
    max_visits: int,
) -> None:
    if tuple(sequences.columns) != SEQUENCE_COLUMNS:
        raise ValueError("recent model sequence columns deviate from the frozen schema")
    forbidden = ("biopsy", "date", "offset", "t0", "first_visit", "absolute")
    if any(token in column.lower() for column in sequences.columns for token in forbidden):
        raise ValueError("recent model sequence table contains a prohibited time/date field")
    for query_id, group in sequences.groupby("query_id", sort=False):
        group = group.sort_values("sequence_index", kind="stable")
        if len(group) > max_visits or list(group["sequence_index"]) != list(range(len(group))):
            raise ValueError(f"query {query_id} violates the 64-visit/index contract")
        gaps = group["intervisit_gap_years"].to_numpy(float)
        if gaps[0] != 0 or not np.isfinite(gaps).all() or np.any(gaps[1:] <= 0):
            raise ValueError(f"query {query_id} violates the inter-visit-gap contract")
        if gaps.sum() > float(window_years) + 1e-8:
            raise ValueError(f"query {query_id} contains data outside the requested recent window")
        if group["CREA"].iloc[-1] != group["CREA"].iloc[-1] or not np.isfinite(float(group["CREA"].iloc[-1])):
            raise ValueError(f"query {query_id} final visit lacks CREA")
        final_companions = group.iloc[-1][["CystatinC", "ALB", "log_PRO24H"]]
        if final_companions.isna().all():
            raise ValueError(f"query {query_id} final visit lacks a companion core marker")
        if group["age_at_query"].nunique(dropna=False) != 1 or group["gender"].nunique(dropna=False) != 1:
            raise ValueError(f"query {query_id} static values are not query-specific constants")
        if not np.isfinite(group[["age_at_query", "gender", "residual_time"]].to_numpy(float)).all():
            raise ValueError(f"query {query_id} contains non-finite model values")
        if (group["residual_time"].to_numpy(float) <= 0).any() or not group["event_status"].isin([0, 1]).all():
            raise ValueError(f"query {query_id} violates the residual ESKD outcome contract")


def load_recent_followup_sequences(processed_dir: str | Path) -> pd.DataFrame:
    path = Path(processed_dir) / "recent_followup_sequences_private.csv"
    frame = pd.read_csv(path, dtype={"patient_id": str}, low_memory=False)
    audit = json.loads((Path(processed_dir) / "snapshot_validation.json").read_text(encoding="utf-8"))
    window = int(audit["window_years"])
    _validate_model_sequence_table(frame, window, MAX_VISITS)
    return frame


def load_recent_followup_patient_outcomes(processed_dir: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(Path(processed_dir) / "recent_followup_patient_outcomes_private.csv", dtype={"patient_id": str})
    required = {"patient_id", "eskd_status"}
    if required.difference(frame.columns) or frame["patient_id"].duplicated().any():
        raise ValueError("recent patient outcome index contract failed")
    return frame


def compute_recent_followup_norm_stats(
    processed_dir: str | Path,
    patient_ids: list[str],
    static_cols: list[str],
    long_cols: list[str],
    patient_multiplicity: dict[str, int] | None = None,
) -> dict:
    """Fit training-fold statistics without accessing date/private audit tables."""
    if tuple(static_cols) != STATIC_COLUMNS or tuple(long_cols) != CORE_MARKERS:
        raise ValueError("recent-follow-up normalization accepts only the locked inputs")
    selected = set(map(str, patient_ids))
    frame = load_recent_followup_sequences(processed_dir)
    frame = frame[frame["patient_id"].astype(str).isin(selected)].copy()
    if frame.empty:
        raise ValueError("selected fold has no recent-follow-up query sequences")
    multiplicity = {patient_id: 1 for patient_id in selected} if patient_multiplicity is None else {
        str(key): int(value) for key, value in patient_multiplicity.items()
    }
    if set(multiplicity) != selected or any(value <= 0 for value in multiplicity.values()):
        raise ValueError("recent bootstrap multiplicity must cover every training patient once")
    # Each patient's valid query anchors jointly receive one unit of weight;
    # within a query, each laboratory row jointly receives one unit for each marker.
    query_counts = frame.groupby("patient_id")["query_id"].transform("nunique").to_numpy(float)
    patient_weight = frame["patient_id"].map(multiplicity).to_numpy(float) / query_counts
    stats: dict[str, dict] = {}
    query_static = frame.sort_values(["query_id", "sequence_index"], kind="stable").drop_duplicates("query_id")
    for column in static_cols:
        values = query_static[["patient_id", column]].dropna()
        query_per_patient = values.groupby("patient_id")[column].transform("size").to_numpy(float)
        weights = values["patient_id"].map(multiplicity).to_numpy(float) / query_per_patient
        numeric = values[column].to_numpy(float)
        mean = float(np.average(numeric, weights=weights))
        std = float(np.sqrt(np.average((numeric - mean) ** 2, weights=weights)))
        stats[column] = {"mean": mean, "std": std if np.isfinite(std) and std > 0 else 1.0,
                         "weighting": "equal_total_weight_per_patient_across_queries", "n_patients": int(values["patient_id"].nunique())}
    for column in long_cols:
        values = frame[["patient_id", "query_id", column]].dropna()
        per_patient_marker_rows = values.groupby("patient_id")[column].transform("size").to_numpy(float)
        weights = values["patient_id"].map(multiplicity).to_numpy(float) / per_patient_marker_rows
        numeric = values[column].to_numpy(float)
        mean = float(np.average(numeric, weights=weights)) if len(values) else 0.0
        std = float(np.sqrt(np.average((numeric - mean) ** 2, weights=weights))) if len(values) else 1.0
        stats[column] = {"mean": mean, "std": std if np.isfinite(std) and std > 0 else 1.0,
                         "weighting": "equal_total_weight_per_patient_across_query_rows", "n_patients": int(values["patient_id"].nunique())}
    return stats


class RecentFollowupDataset(Dataset):
    """Ragged date-free sequences, one sample for each valid query visit."""

    def __init__(
        self,
        *,
        processed_dir: str | Path,
        patient_ids: list[str],
        static_cols: list[str],
        long_cols: list[str],
        max_visits: int,
        time_grid: HalfYearTimeGrid,
        norm_stats: dict,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required only to materialize recent-follow-up training tensors")
        if tuple(static_cols) != STATIC_COLUMNS or tuple(long_cols) != CORE_MARKERS:
            raise ValueError("recent-follow-up dataset accepts only the locked inputs")
        if max_visits != MAX_VISITS:
            raise ValueError("recent-follow-up dataset max_visits is locked to 64")
        frame = load_recent_followup_sequences(processed_dir)
        selected = set(map(str, patient_ids))
        frame = frame[frame["patient_id"].astype(str).isin(selected)].copy()
        self.x_values: list[torch.Tensor] = []
        self.x_missing: list[torch.Tensor] = []
        self.sequence_mask: list[torch.Tensor] = []
        self.target_time: list[torch.Tensor] = []
        self.target_event: list[torch.Tensor] = []
        self.target_bin: list[torch.Tensor] = []
        self.evaluation_time: list[torch.Tensor] = []
        self.evaluation_event: list[torch.Tensor] = []
        self.query_time: list[torch.Tensor] = []
        self.sample_weight: list[torch.Tensor] = []
        self.patient_ids: list[str] = []
        self.query_ids: list[int] = []
        for query_id, group in frame.groupby("query_id", sort=True):
            group = group.sort_values("sequence_index", kind="stable")
            patient_id = str(group["patient_id"].iloc[0])
            if len(group) > max_visits:
                raise ValueError("recent-follow-up dataset refuses a sequence over 64 visits")
            raw_static = group.loc[:, list(static_cols)].iloc[0].to_numpy(float)
            static = np.asarray([
                (value - float(norm_stats[column]["mean"])) / float(norm_stats[column]["std"])
                for column, value in zip(static_cols, raw_static)
            ], dtype=np.float32)
            raw_long = group.loc[:, list(long_cols)].to_numpy(float)
            missing = np.isnan(raw_long).astype(np.float32)
            values = np.empty_like(raw_long, dtype=np.float32)
            for index, column in enumerate(long_cols):
                mean, std = float(norm_stats[column]["mean"]), float(norm_stats[column]["std"])
                values[:, index] = (np.nan_to_num(raw_long[:, index], nan=mean) - mean) / std
            encoded = np.zeros((len(group), 1 + len(static_cols) + len(long_cols)), dtype=np.float32)
            encoded[:, 0] = group["intervisit_gap_years"].to_numpy(float) / float(time_grid.max_time)
            encoded[:, 1:1 + len(static_cols)] = static
            encoded[:, 1 + len(static_cols):] = values
            residual = float(group["residual_time"].iloc[0])
            event = int(group["event_status"].iloc[0])
            target_time, target_event, target_bin = time_grid.administrative_target(
                np.asarray([residual]), np.asarray([event]),
            )
            self.x_values.append(torch.from_numpy(encoded))
            self.x_missing.append(torch.from_numpy(missing))
            self.sequence_mask.append(torch.ones(len(group), dtype=torch.float32))
            self.target_time.append(torch.tensor(float(target_time[0]), dtype=torch.float32))
            self.target_event.append(torch.tensor(int(target_event[0]), dtype=torch.long))
            self.target_bin.append(torch.tensor(int(target_bin[0]), dtype=torch.long))
            self.evaluation_time.append(torch.tensor(residual, dtype=torch.float64))
            self.evaluation_event.append(torch.tensor(event, dtype=torch.long))
            # FormalDynamicDeepHit's clinical-only architecture does not use
            # query_time.  Keep an exact zero so no hidden absolute-time proxy
            # enters the model call or tensor interface.
            self.query_time.append(torch.tensor(0.0, dtype=torch.float32))
            self.sample_weight.append(torch.tensor(1.0, dtype=torch.float32))
            self.patient_ids.append(patient_id)
            self.query_ids.append(int(query_id))
        if not self.patient_ids:
            raise ValueError("selected fold has no eligible recent-follow-up sequences")

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, index: int) -> dict:
        return {
            "x_values": self.x_values[index], "x_missing": self.x_missing[index],
            "sequence_mask": self.sequence_mask[index], "target_time": self.target_time[index],
            "target_event": self.target_event[index], "target_bin": self.target_bin[index],
            "evaluation_time": self.evaluation_time[index], "evaluation_event": self.evaluation_event[index],
            "patient_id": self.patient_ids[index], "query_time": self.query_time[index],
            "sample_weight": self.sample_weight[index],
            "query_id": torch.tensor(self.query_ids[index], dtype=torch.long),
        }
