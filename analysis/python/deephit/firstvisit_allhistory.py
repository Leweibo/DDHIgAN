"""First-valid-visit anchoring for the isolated physv12 DDHIgAN model.

This module intentionally contains no FileMaker access.  It transforms an
approved, private export into the t0-relative model tables and returns only
aggregate audit information to callers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


CORE_MARKERS = ("CREA", "CystatinC", "ALB", "log_PRO24H")
MAX_VISITS = 256


def _median_nonmissing(values: pd.Series):
    observed = values.dropna()
    return np.nan if observed.empty else float(np.median(pd.to_numeric(observed)))


def fixed_span_visit_clusters(
    rows: pd.DataFrame,
    *,
    patient_col: str = "patient_id",
    date_col: str = "visit_date",
    span_days: int = 15,
    marker_cols: tuple[str, ...] = CORE_MARKERS,
) -> pd.DataFrame:
    """Collapse labs using windows fixed to each cluster's first date.

    A row joins the current cluster only when it is no more than ``span_days``
    after that cluster's first date.  The comparison is never made with the
    preceding row, which prevents a chain of 15-day gaps becoming one long
    visit.  The cluster timestamp and each marker are the latest observed
    value/date within the window. Core numeric markers use the project's
    established within-visit median aggregation.
    """
    required = {patient_col, date_col, *marker_cols}
    if missing := required.difference(rows.columns):
        raise ValueError(f"longitudinal source is missing columns: {sorted(missing)}")
    source = rows.copy()
    source[patient_col] = source[patient_col].astype(str)
    source[date_col] = pd.to_datetime(source[date_col], errors="coerce").dt.normalize()
    if source[date_col].isna().any():
        raise ValueError("visit_date must be finite for fixed-span clustering")
    source = source.sort_values([patient_col, date_col], kind="stable")
    output: list[dict] = []
    for patient_id, patient_rows in source.groupby(patient_col, sort=False):
        current: list[int] = []
        cluster_start = None
        for index, row in patient_rows.iterrows():
            date = row[date_col]
            if cluster_start is None or (date - cluster_start).days > span_days:
                if current:
                    cluster = patient_rows.loc[current].sort_values(date_col, kind="stable")
                    output.append({
                        patient_col: patient_id,
                        date_col: cluster[date_col].iloc[-1],
                        **{column: _median_nonmissing(cluster[column]) for column in marker_cols},
                    })
                current = [index]
                cluster_start = date
            else:
                current.append(index)
        if current:
            cluster = patient_rows.loc[current].sort_values(date_col, kind="stable")
            output.append({
                patient_col: patient_id,
                date_col: cluster[date_col].iloc[-1],
                **{column: _median_nonmissing(cluster[column]) for column in marker_cols},
            })
    return pd.DataFrame(output, columns=[patient_col, date_col, *marker_cols])


def select_first_valid_visit(
    clustered: pd.DataFrame,
    *,
    marker_cols: tuple[str, ...] = CORE_MARKERS,
) -> pd.DataFrame:
    """Select first visit with CREA and at least one other core marker."""
    crea, *companions = marker_cols
    valid = clustered[crea].notna() & clustered[list(companions)].notna().any(axis=1)
    candidates = clustered.loc[valid].sort_values(
        ["patient_id", "visit_date"], kind="stable"
    )
    anchors = candidates.drop_duplicates("patient_id", keep="first").copy()
    return anchors.rename(columns={
        "visit_date": "t0_date",
        crea: "t0_CREA",
        companions[0]: "t0_CystatinC",
        companions[1]: "t0_ALB",
        companions[2]: "t0_log_PRO24H",
    })


@dataclass(frozen=True)
class ReanchoredSnapshot:
    baseline: pd.DataFrame
    longitudinal: pd.DataFrame
    outcomes: pd.DataFrame
    offline_anchor: pd.DataFrame
    audit: dict


def build_firstvisit_snapshot(
    baseline: pd.DataFrame,
    longitudinal: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    expected_patients: int = 9948,
    max_visits: int = MAX_VISITS,
    version: str = "ddhigan_firstvisit_allhistory_physv12_20260830",
) -> ReanchoredSnapshot:
    """Build biopsy-date-free, first-valid-visit t0-relative model tables."""
    for name, frame in (("baseline", baseline), ("longitudinal", longitudinal), ("outcomes", outcomes)):
        if "patient_id" not in frame:
            raise ValueError(f"{name} is missing patient_id")
        if frame["patient_id"].astype(str).duplicated().any() and name != "longitudinal":
            raise ValueError(f"{name} contains duplicate patient_id")
    required_baseline = {"patient_id", "age_at_biopsy", "gender", "biopsy_date"}
    required_outcome = {"patient_id", "eskd_time", "eskd_status"}
    if missing := required_baseline.difference(baseline.columns):
        raise ValueError(f"baseline is missing columns: {sorted(missing)}")
    if missing := required_outcome.difference(outcomes.columns):
        raise ValueError(f"outcomes is missing columns: {sorted(missing)}")
    required_longitudinal = {"patient_id", "visit_date", *CORE_MARKERS}
    if missing := required_longitudinal.difference(longitudinal.columns):
        raise ValueError(f"longitudinal is missing columns: {sorted(missing)}")

    for marker in CORE_MARKERS:
        values = pd.to_numeric(longitudinal[marker], errors="coerce")
        if np.isinf(values).any():
            raise ValueError(f"{marker} contains an infinite value")
    clustered = fixed_span_visit_clusters(longitudinal)
    anchors = select_first_valid_visit(clustered)
    base = baseline.copy()
    base["patient_id"] = base["patient_id"].astype(str)
    base["biopsy_date"] = pd.to_datetime(base["biopsy_date"], errors="coerce").dt.normalize()
    anchors["patient_id"] = anchors["patient_id"].astype(str)
    cohort = base.merge(anchors, on="patient_id", how="inner", validate="one_to_one")
    offset_days = (cohort["t0_date"] - cohort["biopsy_date"]).dt.days
    if cohort["biopsy_date"].isna().any() or offset_days.isna().any():
        raise ValueError("finite biopsy_date and t0_date are required offline")
    cohort["age_at_t0"] = pd.to_numeric(cohort["age_at_biopsy"], errors="coerce") + offset_days / 365.0
    if not np.isfinite(cohort["age_at_t0"]).all():
        raise ValueError("age_at_t0 could not be derived")
    cohort["t0_offset_years"] = offset_days / 365.0

    long = clustered.merge(
        cohort[["patient_id", "t0_date"]], on="patient_id", how="inner", validate="many_to_one"
    )
    long = long[long["visit_date"] >= long["t0_date"]].copy()
    long["visit_time"] = (long["visit_date"] - long["t0_date"]).dt.days / 365.0
    long = long.sort_values(["patient_id", "visit_time"], kind="stable")
    visit_counts = long.groupby("patient_id").size()
    observed_max = int(visit_counts.max()) if len(visit_counts) else 0
    if observed_max > max_visits:
        raise ValueError(
            f"snapshot contains {observed_max} visits for one patient; limit is {max_visits}; truncation is prohibited"
        )

    out = outcomes.copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out = out.merge(
        cohort[["patient_id", "t0_offset_years"]], on="patient_id", how="inner", validate="one_to_one"
    )
    out["eskd_time"] = pd.to_numeric(out["eskd_time"], errors="coerce") - out["t0_offset_years"]
    out["eskd_status"] = pd.to_numeric(out["eskd_status"], errors="coerce")
    date_last_col = next((column for column in out.columns if column.lower() == "date_last"), None)
    date_last_audit = {"available": False}
    if date_last_col is not None:
        date_last = pd.to_datetime(out[date_last_col], errors="coerce").dt.normalize()
        t0_lookup = out["patient_id"].map(cohort.set_index("patient_id")["t0_date"])
        derived_censor = (date_last - t0_lookup).dt.days / 365.0
        censored = out["eskd_status"].eq(0) & derived_censor.notna()
        difference_days = (out.loc[censored, "eskd_time"] - derived_censor[censored]) * 365.0
        within_two_days = difference_days.abs() <= 2.0
        date_last_audit = {
            "available": True,
            "censored_patients_checked": int(censored.sum()),
            "within_two_days": int(within_two_days.sum()),
            "mismatch_over_two_days": int((~within_two_days).sum()),
            "endpoint_later_than_date_last": int((difference_days > 2.0).sum()),
            "endpoint_earlier_than_date_last": int((difference_days < -2.0).sum()),
            "median_difference_days": float(np.median(difference_days)),
            "minimum_difference_days": float(np.min(difference_days)),
            "maximum_difference_days": float(np.max(difference_days)),
            "action": "retain manually rechecked ESKD time; expose discrepancy for human confirmation",
        }
    if not bool(out["eskd_status"].isin([0, 1]).all()) or (~np.isfinite(out["eskd_time"])).any():
        raise ValueError("rechecked ESKD outcome contract failed")
    if (out["eskd_time"] <= 0).any():
        raise ValueError("t0 is not strictly before every ESKD/censoring time")

    ids = set(cohort["patient_id"])
    if len(ids) != expected_patients or ids != set(long["patient_id"]) or ids != set(out["patient_id"]):
        raise ValueError(
            f"t0 eligibility/key gate failed: expected {expected_patients}, observed {len(ids)}"
        )
    baseline_out = cohort[[
        "patient_id", "age_at_t0", "gender", "t0_CREA", "t0_CystatinC",
        "t0_ALB", "t0_log_PRO24H",
    ]].sort_values("patient_id")
    longitudinal_out = long[["patient_id", "visit_time", *CORE_MARKERS]].copy()
    outcome_cols = [column for column in out.columns if column not in {"t0_offset_years", date_last_col}]
    outcomes_out = out[outcome_cols].sort_values("patient_id")
    offline_anchor = cohort[["patient_id", "biopsy_date", "t0_date", "t0_offset_years"]].sort_values("patient_id")
    eligible = long.merge(
        out[["patient_id", "eskd_time"]], on="patient_id", how="inner", validate="many_to_one"
    )
    eligible = eligible[eligible["visit_time"] < eligible["eskd_time"]]
    max_supported_query = float(eligible["visit_time"].max())
    audit = {
        "status": "READY_FOR_HUMAN_CONFIRMATION",
        "version": version,
        "patients": len(ids),
        "longitudinal_rows": int(len(longitudinal_out)),
        "max_visits_per_patient": observed_max,
        "patients_over_256_visits": int((visit_counts > MAX_VISITS).sum()),
        "t0_definition": "first fixed-span visit with CREA and >=1 of CystatinC/ALB/log_PRO24H",
        "cluster_contract": "non-transitive fixed 15-day span; timestamp=cluster end",
        "date_fields_in_model_tables": False,
        "offline_anchor_is_private_and_excluded_from_model": True,
        "log_PRO24H_contract": "strict natural log; zero and missing are NA",
        "max_supported_query_time": max_supported_query,
        "date_last_crosscheck": date_last_audit,
        "training_started": False,
    }
    return ReanchoredSnapshot(baseline_out, longitudinal_out, outcomes_out, offline_anchor, audit)
