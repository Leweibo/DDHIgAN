import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import tempfile

try:
    from utils.time_grid import HalfYearTimeGrid
except ModuleNotFoundError:
    from .time_grid import HalfYearTimeGrid


def load_longitudinal_data(processed_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load longitudinal and outcome data from CSV."""
    processed_dir = Path(processed_dir)
    long_df = pd.read_csv(processed_dir / "igan_longitudinal.csv", low_memory=False)
    baseline_path = processed_dir / "igan_baseline.csv"
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path, low_memory=False)
        # Operational convention: an undocumented history of hypertension or
        # diabetes is treated as absent, not as an independent "unknown"
        # clinical state. This keeps the deployment form binary.
        for column in ("baseline_hypertension", "baseline_diabetes"):
            if column in baseline_df.columns:
                baseline_df[column] = pd.to_numeric(
                    baseline_df[column], errors="coerce"
                ).fillna(0.0)
        baseline_cols = [
            col for col in baseline_df.columns
            if col not in long_df.columns or col == "patient_id"
        ]
        long_df = long_df.merge(
            baseline_df[baseline_cols],
            on="patient_id",
            how="left",
            validate="many_to_one",
        )
    # Deployment-facing chronological age.  It is deterministically derived
    # from date of birth in the future web interface; the research dataset uses
    # the equivalent biopsy-age plus elapsed-visit-time representation.
    if "age_at_biopsy" in long_df.columns and "visit_time" in long_df.columns:
        long_df["age_at_visit"] = (
            pd.to_numeric(long_df["age_at_biopsy"], errors="coerce")
            + pd.to_numeric(long_df["visit_time"], errors="coerce")
        )
    outcomes_df = pd.read_csv(processed_dir / "igan_outcomes.csv")
    return long_df, outcomes_df


def filter_valid_event_outcomes(
    outcomes_df: pd.DataFrame,
    time_col: str = "event_time",
    status_col: str = "event_status",
) -> pd.DataFrame:
    """Return outcomes with finite time and binary event status."""
    required = {"patient_id", time_col, status_col}
    missing = required.difference(outcomes_df.columns)
    if missing:
        raise ValueError(f"outcomes_df is missing columns: {sorted(missing)}")
    result = outcomes_df.copy()
    result[time_col] = pd.to_numeric(result[time_col], errors="coerce")
    result[status_col] = pd.to_numeric(result[status_col], errors="coerce")
    valid = (
        np.isfinite(result[time_col].to_numpy(dtype=float))
        & np.isfinite(result[status_col].to_numpy(dtype=float))
        & result[status_col].isin([0, 1])
    )
    return result.loc[valid].copy()


def reanchor_to_discharge(
    longitudinal_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    discharge_dates: pd.DataFrame,
    *,
    visit_date_col: str = "visit_date",
    discharge_date_col: str = "discharge_date",
    biopsy_date_col: str = "biopsy_date",
    outcome_time_cols: tuple[str, ...] = ("eskd_time", "drop50_time"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return new discharge-relative inputs without mutating biopsy-relative data.

    Outcome times are converted from the biopsy-relative convention by subtracting
    the patient-specific discharge-minus-biopsy offset. This is algebraically
    equivalent to recomputing from event/censor dates while preserving the
    rechecked endpoint contract.
    """
    required = {"patient_id", visit_date_col}
    if missing := required.difference(longitudinal_df.columns):
        raise ValueError(f"longitudinal_df is missing columns: {sorted(missing)}")
    if missing := {"patient_id", discharge_date_col}.difference(discharge_dates.columns):
        raise ValueError(f"discharge_dates is missing columns: {sorted(missing)}")
    if discharge_dates["patient_id"].astype(str).duplicated().any():
        raise ValueError("discharge_dates contains duplicate patient_id")
    result = longitudinal_df.copy()
    anchors = discharge_dates[["patient_id", discharge_date_col]].copy()
    result = result.merge(anchors, on="patient_id", how="left", validate="many_to_one")
    visit = pd.to_datetime(result[visit_date_col], errors="coerce")
    discharge = pd.to_datetime(result[discharge_date_col], errors="coerce")
    if visit.isna().any() or discharge.isna().any():
        raise ValueError("discharge-anchor conversion requires finite visit and discharge dates")
    result["visit_time"] = ((visit - discharge).dt.total_seconds() / (365.25 * 86400)).astype(float)
    if biopsy_date_col not in discharge_dates.columns:
        raise ValueError(f"discharge_dates is missing {biopsy_date_col}")
    outcomes = outcomes_df.copy().merge(discharge_dates[["patient_id", discharge_date_col, biopsy_date_col]], on="patient_id", how="left", validate="many_to_one")
    outcome_offset = ((pd.to_datetime(outcomes[discharge_date_col]) - pd.to_datetime(outcomes[biopsy_date_col])).dt.total_seconds() / (365.25 * 86400))
    if not np.isfinite(outcome_offset.to_numpy(dtype=float)).all():
        raise ValueError("discharge-anchor conversion requires finite biopsy dates")
    for column in outcome_time_cols:
        if column in outcomes.columns:
            outcomes[column] = pd.to_numeric(outcomes[column], errors="coerce") - outcome_offset
    return result.drop(columns=[discharge_date_col]), outcomes.drop(columns=[discharge_date_col, biopsy_date_col])


def get_fold_split(splits_dir: str | Path, fold: int, subset: str) -> list[str]:
    """Return list of patient_ids for a given fold and subset ('train' or 'test')."""
    splits_dir = Path(splits_dir)
    fpath = splits_dir / f"fold_{fold}_{subset}.csv"
    df = pd.read_csv(fpath)
    return df["patient_id"].astype(str).tolist()


def anchor_nearest_visit_at_t0(
    patient_rows: pd.DataFrame,
    min_visit_time: float | None,
) -> pd.DataFrame:
    """Use the visit closest to biopsy as the actual t0 visit.

    The selected row is re-timestamped to zero.  With a finite
    ``min_visit_time``, earlier rows are excluded as before; with ``None``,
    they are retained so an AllHistory model can share the exact same t0
    anchor while adding the remaining pre-biopsy history.  In an exact-distance
    tie, the post-biopsy visit is preferred.
    """
    rows = patient_rows.sort_values("visit_time").copy()
    if rows.empty or (min_visit_time is not None and float(min_visit_time) < 0):
        return rows.reset_index(drop=True)

    times = pd.to_numeric(rows["visit_time"], errors="coerce")
    valid = np.isfinite(times.to_numpy(dtype=float))
    if not valid.any():
        return rows.iloc[0:0].copy().reset_index(drop=True)
    candidates = rows.loc[valid].copy()
    candidate_times = times.loc[valid].to_numpy(dtype=float)
    # lexsort's final key is primary: absolute distance, then prefer >= 0.
    choice = np.lexsort((candidate_times, candidate_times < 0, np.abs(candidate_times)))[0]
    anchor_index = candidates.index[int(choice)]
    anchor = rows.loc[[anchor_index]].copy()
    anchor["visit_time"] = 0.0

    retained = rows.drop(index=anchor_index)
    if min_visit_time is not None:
        retained = retained[times.loc[retained.index] >= float(min_visit_time)]
    return pd.concat([anchor, retained], ignore_index=True).sort_values("visit_time").reset_index(drop=True)


def compute_dynamic_norm_stats(
    long_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    static_cols: list[str],
    long_cols: list[str],
    patient_ids: list[str],
    min_visit_time: float | None = None,
    anchor_nearest_t0: bool = False,
    max_visit_time: float | None = None,
    patient_multiplicity: dict[str, int] | None = None,
) -> dict:
    """Fit patient-balanced normalization using only pre-outcome training data.

    Static covariates contribute once per patient. Longitudinal covariates give
    every patient equal total weight, rather than giving frequent attenders a
    larger influence merely because they have more recorded visits.
    """
    pid_set = set(map(str, patient_ids))
    multiplicity = (
        {patient_id: 1 for patient_id in pid_set}
        if patient_multiplicity is None
        else {str(key): int(value) for key, value in patient_multiplicity.items()}
    )
    if set(multiplicity) != pid_set or any(value <= 0 for value in multiplicity.values()):
        raise ValueError(
            "patient_multiplicity must contain every selected patient exactly once with positive counts"
        )
    long_sub = long_df[long_df["patient_id"].astype(str).isin(pid_set)].copy()
    outcome_sub = outcomes_df[
        outcomes_df["patient_id"].astype(str).isin(pid_set)
    ][["patient_id", "event_time"]].copy()
    outcome_sub["event_status"] = 0
    outcome_sub = filter_valid_event_outcomes(outcome_sub)
    outcome_sub = outcome_sub[["patient_id", "event_time"]]
    outcome_sub["patient_id"] = outcome_sub["patient_id"].astype(str)
    long_sub["patient_id"] = long_sub["patient_id"].astype(str)
    long_sub = long_sub.merge(outcome_sub, on="patient_id", how="inner")
    long_sub = long_sub[long_sub["visit_time"] <= long_sub["event_time"]]
    if anchor_nearest_t0:
        long_sub = pd.concat(
            [
                anchor_nearest_visit_at_t0(group, min_visit_time)
                for _, group in long_sub.groupby("patient_id", sort=False)
            ],
            ignore_index=True,
        ) if not long_sub.empty else long_sub
    elif min_visit_time is not None:
        long_sub = long_sub[long_sub["visit_time"] >= float(min_visit_time)]
    if max_visit_time is not None:
        long_sub = long_sub[long_sub["visit_time"] <= float(max_visit_time)]

    stats = {}
    for column in static_cols:
        static_frame = (
            long_sub.sort_values(["patient_id", "visit_time"])
            .drop_duplicates("patient_id", keep="first")[["patient_id", column]]
            .dropna()
        )
        numeric = static_frame[column].to_numpy(dtype=float)
        weights = static_frame["patient_id"].map(multiplicity).to_numpy(dtype=float)
        mean = float(np.average(numeric, weights=weights)) if len(numeric) else 0.0
        variance = (
            float(np.average((numeric - mean) ** 2, weights=weights))
            if len(numeric) else 1.0
        )
        std = float(np.sqrt(variance))
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        stats[column] = {
            "mean": mean, "std": std,
            "weighting": (
                "bootstrap_multiplicity_per_patient"
                if patient_multiplicity is not None
                else "one_observation_per_patient"
            ),
            "n_patients": int(len(static_frame)),
        }
        if patient_multiplicity is not None:
            stats[column]["bootstrap_slots"] = int(weights.sum()) if len(weights) else 0
    for column in long_cols:
        values = long_sub[["patient_id", column]].dropna()
        if values.empty:
            mean, std, n_patients = 0.0, 1.0, 0
        else:
            counts = values.groupby("patient_id")[column].transform("size").to_numpy(dtype=float)
            weights = (
                values["patient_id"].map(multiplicity).to_numpy(dtype=float) / counts
            )
            numeric = values[column].to_numpy(dtype=float)
            mean = float(np.average(numeric, weights=weights))
            variance = float(np.average((numeric - mean) ** 2, weights=weights))
            std = float(np.sqrt(variance))
            if not np.isfinite(std) or std <= 0:
                std = 1.0
            n_patients = int(values["patient_id"].nunique())
        stats[column] = {
            "mean": mean, "std": std,
            "weighting": (
                "bootstrap_multiplicity_equal_within_patient"
                if patient_multiplicity is not None
                else "equal_total_weight_per_patient"
            ),
            "n_patients": n_patients,
        }
        if patient_multiplicity is not None:
            stats[column]["bootstrap_slots"] = int(sum(
                multiplicity[str(patient_id)] for patient_id in values["patient_id"].unique()
            )) if not values.empty else 0
    return stats


def save_norm_stats(stats: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def load_norm_stats(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def build_dynamic_prefix_inputs(
    long_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    patient_ids: list[str],
    static_cols: list[str],
    long_cols: list[str],
    max_visits: int,
    time_grid: HalfYearTimeGrid,
    evaluation_landmarks: list[float] | None = None,
    norm_stats: dict | None = None,
    include_actual_queries: bool = True,
    max_query_time: float | None = None,
    min_history_time: float | None = None,
    anchor_nearest_t0: bool = False,
    append_missing_query_row: bool = True,
    evaluation_history_window: float = 0.0,
    include_time_delta: bool = False,
    encode_first_visit_from_biopsy: bool = False,
    ragged: bool = False,
    reject_excess_visits: bool = False,
    evaluation_landmarks_by_patient: dict[str, list[float]] | None = None,
    input_time_scale: float | None = None,
    preserve_source_time: bool = False,
    actual_query_end_tolerance: float = 0.0,
) -> dict:
    """Build Dynamic-DeepHit prefixes at prespecified landmarks or visits."""
    input_time_scale = float(time_grid.max_time if input_time_scale is None else input_time_scale)
    if not np.isfinite(input_time_scale) or input_time_scale <= 0:
        raise ValueError("input_time_scale must be finite and positive")
    if preserve_source_time and evaluation_history_window != 0:
        raise ValueError("source-precision queries require strict history cutoff")
    if not np.isfinite(actual_query_end_tolerance) or actual_query_end_tolerance < 0:
        raise ValueError("actual_query_end_tolerance must be finite and nonnegative")
    evaluation_landmarks = evaluation_landmarks or []
    if preserve_source_time:
        queries = list(evaluation_landmarks) + [q for qs in (evaluation_landmarks_by_patient or {}).values() for q in qs]
        if any(not np.isfinite(q) or q < 0 or q > 5 for q in queries):
            raise ValueError("source-precision evaluation queries must lie in [0, 5]")
    pid_set = set(map(str, patient_ids))
    long_sub = long_df[long_df["patient_id"].astype(str).isin(pid_set)].copy()
    out_sub = outcomes_df[outcomes_df["patient_id"].astype(str).isin(pid_set)].copy()
    out_sub = filter_valid_event_outcomes(out_sub)
    out_index = out_sub.set_index(out_sub["patient_id"].astype(str))

    if norm_stats is None:
        norm_stats = {}
        for column in static_cols + long_cols:
            values = long_sub[column].dropna()
            norm_stats[column] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
            }

    static_dim = len(static_cols)
    long_dim = len(long_cols)
    value_dim = 1 + static_dim + long_dim
    samples = []
    residual_times = []
    observed_events = []
    sample_weights = []
    for patient_id, patient_rows in long_sub.groupby("patient_id", sort=False):
        pid = str(patient_id)
        if pid not in out_index.index:
            continue
        patient_rows = patient_rows.sort_values("visit_time").reset_index(drop=True)
        if anchor_nearest_t0:
            patient_rows = anchor_nearest_visit_at_t0(patient_rows, min_history_time)
        elif min_history_time is not None:
            patient_rows = patient_rows[
                patient_rows["visit_time"] >= float(min_history_time)
            ].reset_index(drop=True)
        if patient_rows.empty:
            continue
        outcome = out_index.loc[pid]
        observed_time = float(outcome["event_time"])
        observed_event = int(outcome["event_status"])
        visit_values = patient_rows["visit_time"].to_numpy(
            dtype=np.float64 if preserve_source_time else np.float32
        )

        if static_dim:
            raw_static = patient_rows[static_cols].to_numpy(dtype=np.float32)
            static_values = np.zeros((len(patient_rows), static_dim), dtype=np.float32)
            for index, column in enumerate(static_cols):
                mean = float(norm_stats[column]["mean"])
                std = float(norm_stats[column]["std"])
                values = np.nan_to_num(raw_static[:, index], nan=mean).astype(np.float32)
                static_values[:, index] = (values - mean) / std if std > 0 else values - mean
        else:
            static_values = np.zeros((len(patient_rows), 0), dtype=np.float32)

        raw_long = patient_rows[long_cols].to_numpy(dtype=np.float32)
        long_missing_values = np.isnan(raw_long).astype(np.float32)
        long_values = np.zeros((len(patient_rows), long_dim), dtype=np.float32)
        for index, column in enumerate(long_cols):
            mean = float(norm_stats[column]["mean"])
            std = float(norm_stats[column]["std"])
            values = np.nan_to_num(raw_long[:, index], nan=mean).astype(np.float32)
            long_values[:, index] = (values - mean) / std if std > 0 else values - mean

        long_delta_values = np.zeros_like(long_values)
        long_carried_values = np.zeros_like(long_values)
        if include_time_delta:
            last_time = np.full(long_dim, np.nan, dtype=np.float32)
            last_value = np.zeros(long_dim, dtype=np.float32)
            for row_index, visit_time in enumerate(visit_values):
                observed = long_missing_values[row_index] == 0
                last_time[observed] = visit_time
                last_value[observed] = long_values[row_index, observed]
                long_carried_values[row_index] = last_value
                known = np.isfinite(last_time)
                long_delta_values[row_index, known] = np.maximum(
                    visit_time - last_time[known], 0.0
                )

        # One patient-level encoded backing array. Ragged actual-query prefixes
        # below are basic-slice views of this array, so histories are not copied
        # into a fixed n_prefixes x max_visits tensor before batching.
        patient_encoded = np.zeros((len(patient_rows), value_dim), dtype=np.float32)
        patient_delta = np.zeros(len(patient_rows), dtype=np.float32)
        if encode_first_visit_from_biopsy and len(patient_rows):
            patient_delta[0] = visit_values[0]
        if len(patient_rows) > 1:
            patient_delta[1:] = np.diff(visit_values)
        patient_encoded[:, 0] = patient_delta / input_time_scale
        if static_dim:
            patient_encoded[:, 1:1 + static_dim] = static_values
        if long_dim:
            patient_encoded[:, 1 + static_dim:] = (
                long_carried_values if include_time_delta else long_values
            )

        actual_queries = []
        if include_actual_queries:
            actual_queries = visit_values[
                (visit_values >= 0)
                # Exclude numerical zero-length terminal prefixes without
                # rounding source query/history times or changing evaluation risk sets.
                & (visit_values < observed_time - actual_query_end_tolerance)
                & (
                    True
                    if max_query_time is None
                    else visit_values <= float(max_query_time)
                )
            ].astype(float).tolist()
        patient_landmarks = list(evaluation_landmarks)
        if evaluation_landmarks_by_patient is not None:
            patient_landmarks.extend(evaluation_landmarks_by_patient.get(pid, []))
        query_candidates = sorted(
            float(query)
            for query in actual_queries + patient_landmarks
            if 0 <= float(query) < observed_time
        )
        query_times = []
        for query in query_candidates:
            if not query_times or not np.isclose(query, query_times[-1]):
                query_times.append(query)

        for query_time in query_times:
            history_cutoff_time = float(query_time) + float(evaluation_history_window)
            if history_cutoff_time >= observed_time:
                history_cutoff_time = float(np.nextafter(observed_time, -np.inf))
            end = int(np.searchsorted(visit_values, history_cutoff_time, side="right"))
            if reject_excess_visits and end > max_visits:
                raise ValueError(
                    f"patient {pid} has {end} visits at query {query_time}; "
                    f"limit is {max_visits}; truncation is prohibited"
                )
            exact_query = end > 0 and bool(np.any(np.isclose(visit_values[:end], query_time)))
            if exact_query:
                start = 0 if ragged else max(0, end - max_visits)
                sequence_times = visit_values[start:end]
                sequence_static = static_values[start:end]
                sequence_long = (
                    long_carried_values if include_time_delta else long_values
                )[start:end]
                sequence_missing = long_missing_values[start:end]
                sequence_delta = long_delta_values[start:end]
                sequence_encoded = patient_encoded[start:end] if ragged else None
            elif end == 0:
                if preserve_source_time:
                    raise ValueError("query has no observed history; synthetic rows are forbidden")
                sequence_times = np.asarray([query_time], dtype=np.float32)
                sequence_static = static_values[[0]]
                sequence_long = np.zeros((1, long_dim), dtype=np.float32)
                sequence_missing = np.ones((1, long_dim), dtype=np.float32)
                sequence_delta = np.zeros((1, long_dim), dtype=np.float32)
                sequence_encoded = None
            elif not append_missing_query_row:
                start = 0 if ragged else max(0, end - max_visits)
                sequence_times = visit_values[start:end]
                sequence_static = static_values[start:end]
                sequence_long = (
                    long_carried_values if include_time_delta else long_values
                )[start:end]
                sequence_missing = long_missing_values[start:end]
                sequence_delta = long_delta_values[start:end]
                sequence_encoded = patient_encoded[start:end] if ragged else None
            else:
                if reject_excess_visits and end + 1 > max_visits:
                    raise ValueError(
                        f"patient {pid} requires {end + 1} visits at query {query_time}; "
                        f"limit is {max_visits}; truncation is prohibited"
                    )
                start = 0 if ragged else max(0, end - (max_visits - 1))
                sequence_times = np.concatenate(
                    [visit_values[start:end], np.asarray([query_time], dtype=np.float32)]
                )
                sequence_static = np.vstack([static_values[start:end], static_values[[end - 1]]])
                source_long = long_carried_values if include_time_delta else long_values
                query_long = (
                    long_carried_values[[end - 1]]
                    if include_time_delta else np.zeros((1, long_dim), dtype=np.float32)
                )
                sequence_long = np.vstack([source_long[start:end], query_long])
                sequence_missing = np.vstack(
                    [long_missing_values[start:end], np.ones((1, long_dim), dtype=np.float32)]
                )
                query_delta = np.zeros((1, long_dim), dtype=np.float32)
                if include_time_delta:
                    for column_index in range(long_dim):
                        observed_rows = np.flatnonzero(
                            long_missing_values[:end, column_index] == 0
                        )
                        if observed_rows.size:
                            query_delta[0, column_index] = max(
                                query_time - float(visit_values[observed_rows[-1]]), 0.0
                            )
                sequence_delta = np.vstack([long_delta_values[start:end], query_delta])
                sequence_encoded = None

            samples.append(
                {
                    "patient_id": pid,
                    "query_time": query_time,
                    "last_observation_time": float(visit_values[end - 1]) if end else np.nan,
                    "visit_time": sequence_times,
                    "static_values": sequence_static,
                    "long_values": sequence_long,
                    "long_missing": sequence_missing,
                    "long_delta": sequence_delta,
                    "encoded_values": sequence_encoded,
                }
            )
            residual_times.append(observed_time - query_time)
            observed_events.append(observed_event)
            sample_weights.append(1.0 / len(query_times))

    n_samples = len(samples)
    if ragged:
        x_values, x_missing, sequence_mask, x_time, x_delta = [], [], [], [], []
        for sample in samples:
            length = len(sample["visit_time"])
            values = sample["encoded_values"]
            if values is None:
                values = np.zeros((length, value_dim), dtype=np.float32)
                delta = np.zeros(length, dtype=np.float32)
                if encode_first_visit_from_biopsy and length:
                    delta[0] = float(sample["visit_time"][0])
                if length > 1:
                    delta[1:] = np.diff(sample["visit_time"]).astype(np.float32)
                values[:, 0] = delta / input_time_scale
                if static_dim:
                    values[:, 1:1 + static_dim] = sample["static_values"]
                if long_dim:
                    values[:, 1 + static_dim:] = sample["long_values"]
            x_values.append(values)
            x_missing.append(sample["long_missing"].astype(np.float32, copy=False))
            sequence_mask.append(np.ones(length, dtype=np.float32))
            x_time.append(np.asarray(sample["visit_time"], dtype=np.float32))
            x_delta.append(sample["long_delta"].astype(np.float32, copy=False))
    else:
        x_values = np.zeros((n_samples, max_visits, value_dim), dtype=np.float32)
        x_missing = np.zeros((n_samples, max_visits, long_dim), dtype=np.float32)
        sequence_mask = np.zeros((n_samples, max_visits), dtype=np.float32)
        x_time = np.zeros((n_samples, max_visits), dtype=np.float32)
        x_delta = np.zeros((n_samples, max_visits, long_dim), dtype=np.float32)
        for index, sample in enumerate(samples):
            length = len(sample["visit_time"])
            delta = np.zeros(length, dtype=np.float32)
            if encode_first_visit_from_biopsy and length:
                delta[0] = float(sample["visit_time"][0])
            if length > 1:
                delta[1:] = np.diff(sample["visit_time"]).astype(np.float32)
            delta /= input_time_scale
            x_values[index, :length, 0] = delta
            if static_dim:
                x_values[index, :length, 1:1 + static_dim] = sample["static_values"]
            if long_dim:
                x_values[index, :length, 1 + static_dim:] = sample["long_values"]
                x_missing[index, :length] = sample["long_missing"]
                x_delta[index, :length] = sample["long_delta"]
            x_time[index, :length] = sample["visit_time"]
            sequence_mask[index, :length] = 1.0
    # Keep evaluation labels at source precision; model targets remain float32.
    residual_array = np.asarray(residual_times, dtype=np.float64)
    event_array = np.asarray(observed_events, dtype=np.int64)
    target_time, target_event, target_bin = time_grid.administrative_target(
        residual_array, event_array
    )

    result = {
        "x_values": x_values,
        "x_missing": x_missing,
        "sequence_mask": sequence_mask,
        "target_time": target_time.astype(np.float32),
        "target_event": target_event.astype(np.int64),
        "target_bin": target_bin.astype(np.int64),
        "evaluation_time": residual_array,
        "evaluation_event": event_array,
        "patient_id": [sample["patient_id"] for sample in samples],
        "query_time": np.asarray([sample["query_time"] for sample in samples],
                                 dtype=np.float64 if preserve_source_time else np.float32),
        "last_observation_time": np.asarray(
            [sample["last_observation_time"] for sample in samples], dtype=np.float64),
        "sample_weight": np.asarray(sample_weights, dtype=np.float32),
    }
    if include_time_delta:
        result["x_time"] = x_time
        result["x_delta"] = x_delta
    return result
