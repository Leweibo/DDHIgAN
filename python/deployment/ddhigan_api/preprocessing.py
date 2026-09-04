from __future__ import annotations

import math

import numpy as np

from .schemas import PredictionRequest, PredictionRequestV2, PredictionRequestV3


STATIC_COLUMNS = (
    "age_at_biopsy", "gender", "baseline_CREA", "baseline_CystatinC",
    "baseline_ALB", "baseline_log_PRO24H",
)
LONG_COLUMNS = ("CREA", "CystatinC", "ALB", "log_PRO24H")
V2_STATIC_COLUMNS = (
    "age_at_t0", "gender", "t0_CREA", "t0_CystatinC", "t0_ALB", "t0_log_PRO24H",
)
V3_STATIC_COLUMNS = ("age_at_query", "gender")
LATE_FOLLOWUP_EXTRAPOLATION_WARNING = (
    "Query time is more than 5 years after biopsy; this prediction is a research "
    "extrapolation beyond the model's 0-5-year training-query and internal-validation range."
)


def late_followup_extrapolation_warning(request) -> str | None:
    if request.schema_version == "1.0" and request.query_time_years > 5.0 + 1e-8:
        return LATE_FOLLOWUP_EXTRAPOLATION_WARNING
    return None


def _log_protein(value: float | None, *, strict: bool = False) -> float | None:
    if value is None:
        return None
    if strict:
        return None if value == 0 else math.log(value)
    # The deployed biopsy-anchored v1 model was trained with this offset.
    return math.log(value + 0.001)


def _normalize(value: float | None, column: str, stats: dict) -> tuple[float, bool]:
    missing = value is None
    raw = float(stats[column]["mean"]) if missing else float(value)
    return (raw - float(stats[column]["mean"])) / float(stats[column]["std"]), missing


def prepare_model_inputs(request: PredictionRequest | PredictionRequestV2 | PredictionRequestV3, norm_stats: dict, max_visits: int = 30):
    if max_visits <= 0:
        raise ValueError("model max_visits must be positive")
    is_v2 = request.schema_version == "2.0"
    is_v3 = request.schema_version == "3.0"
    strict_protein_log = is_v2 or is_v3
    static_columns = V3_STATIC_COLUMNS if is_v3 else V2_STATIC_COLUMNS if is_v2 else STATIC_COLUMNS
    static_raw = (
        (request.static.age_at_query_years, 1.0 if request.static.sex == "female" else 0.0)
        if is_v3 else
        (
            request.static.age_at_t0_years if is_v2 else request.static.age_at_biopsy_years,
            1.0 if request.static.sex == "female" else 0.0,
            request.static.creatinine_mg_dl,
            request.static.cystatin_c_mg_l,
            request.static.albumin_g_l,
            _log_protein(request.static.proteinuria_g_24h, strict=strict_protein_log),
        )
    )
    static_values, missing_fields = [], []
    for column, value in zip(static_columns, static_raw):
        normalized, missing = _normalize(value, column, norm_stats)
        static_values.append(normalized)
        if missing:
            missing_fields.append(f"static.{column}")
    received_visits = list(request.visits)
    first_encoded_index = max(0, len(received_visits) - max_visits)
    encoded_visits = received_visits[first_encoded_index:]
    n = len(encoded_visits)
    value_dim = 1 + len(static_columns) + len(LONG_COLUMNS)
    x_values = np.zeros((1, max_visits, value_dim), dtype=np.float32)
    x_missing = np.zeros((1, max_visits, len(LONG_COLUMNS)), dtype=np.float32)
    sequence_mask = np.zeros((1, max_visits), dtype=np.float32)
    if is_v3:
        full_deltas = np.asarray(
            [visit.intervisit_gap_years for visit in received_visits], dtype=np.float32
        )
    else:
        times = np.asarray([visit.time_years for visit in received_visits], dtype=np.float32)
        full_deltas = np.zeros(len(received_visits), dtype=np.float32)
        if len(received_visits) > 1:
            full_deltas[1:] = np.diff(times)
    deltas = full_deltas[first_encoded_index:]
    x_values[0, :n, 0] = deltas / 10.0
    x_values[0, :n, 1:1 + len(static_columns)] = np.asarray(static_values, dtype=np.float32)
    for row, visit in enumerate(encoded_visits):
        raw = (
            visit.creatinine_mg_dl, visit.cystatin_c_mg_l,
            visit.albumin_g_l,
            _log_protein(visit.proteinuria_g_24h, strict=strict_protein_log),
        )
        for column_index, (column, value) in enumerate(zip(LONG_COLUMNS, raw)):
            normalized, missing = _normalize(value, column, norm_stats)
            x_values[0, row, 1 + len(static_columns) + column_index] = normalized
            x_missing[0, row, column_index] = float(missing)
            if missing:
                missing_fields.append(f"visits[{first_encoded_index + row}].{column}")
    sequence_mask[0, :n] = 1.0
    return {
        "x_values": x_values,
        "x_missing": x_missing,
        "sequence_mask": sequence_mask,
        "query_time": np.asarray([0.0 if is_v3 else request.query_time_years], dtype=np.float32),
        "missing_fields": sorted(set(missing_fields)),
        "history_processing": {
            "received_visits": len(received_visits),
            "encoded_visits": len(encoded_visits),
            "omitted_visits": first_encoded_index,
            "rule": "most_recent_30_visits_before_or_at_query",
        },
    }


def distribution_warnings(request: PredictionRequest | PredictionRequestV2 | PredictionRequestV3, ranges: dict) -> list[str]:
    is_v2 = request.schema_version == "2.0"
    is_v3 = request.schema_version == "3.0"
    raw = {
        "gender": [1.0 if request.static.sex == "female" else 0.0],
        "CREA": [visit.creatinine_mg_dl for visit in request.visits],
        "CystatinC": [visit.cystatin_c_mg_l for visit in request.visits],
        "ALB": [visit.albumin_g_l for visit in request.visits],
        "log_PRO24H": [
            _log_protein(visit.proteinuria_g_24h, strict=is_v2 or is_v3)
            for visit in request.visits
        ],
    }
    if is_v3:
        raw["age_at_query"] = [request.static.age_at_query_years]
    else:
        raw.update({
            ("age_at_t0" if is_v2 else "age_at_biopsy"): [
                request.static.age_at_t0_years if is_v2 else request.static.age_at_biopsy_years
            ],
            ("t0_CREA" if is_v2 else "baseline_CREA"): [request.static.creatinine_mg_dl],
            ("t0_CystatinC" if is_v2 else "baseline_CystatinC"): [request.static.cystatin_c_mg_l],
            ("t0_ALB" if is_v2 else "baseline_ALB"): [request.static.albumin_g_l],
            ("t0_log_PRO24H" if is_v2 else "baseline_log_PRO24H"): [
                _log_protein(request.static.proteinuria_g_24h, strict=is_v2)
            ],
        })
    warnings = []
    for column, values in raw.items():
        lower, upper = ranges[column]["p01"], ranges[column]["p99"]
        if any(value is not None and (value < lower or value > upper) for value in values):
            warnings.append(f"{column} outside training p01-p99 reference range")
    return warnings
