"""Named clinical predictor sets for reproducible DDHIgAN comparisons.

Dynamic biomarkers are atomic groups: a group is represented by its biopsy
baseline value in static inputs and its whole observed trajectory in dynamic
inputs.  This prevents a selected biomarker from contributing only one of the
two information sources.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalVariableSet:
    name: str
    cox_static_cols: tuple[str, ...]
    dynamic_cols: tuple[str, ...]
    baseline_fallback: dict[str, str]

    @property
    def ddh_static_cols(self) -> tuple[str, ...]:
        # Age is supplied to DDH at every observed visit as age_at_visit.
        # Its baseline representation is therefore intentionally not repeated
        # in static inputs.
        return (
            *self.cox_static_cols,
            *[
                self.baseline_fallback[col]
                for col in self.dynamic_cols
                if col != "age_at_visit"
            ],
        )


_MINIMAL_CORE = ClinicalVariableSet(
    name="minimal_core",
    cox_static_cols=("age_at_biopsy", "gender"),
    dynamic_cols=("CREA", "CystatinC", "ALB", "log_PRO24H"),
    baseline_fallback={
        "CREA": "baseline_CREA",
        "CystatinC": "baseline_CystatinC",
        "ALB": "baseline_ALB",
        "log_PRO24H": "baseline_log_PRO24H",
    },
)

_ROUTINE9 = ClinicalVariableSet(
    name="routine9",
    cox_static_cols=("gender", "baseline_hypertension", "baseline_diabetes"),
    dynamic_cols=("CREA", "CystatinC", "ALB", "log_PRO24H", "UA", "age_at_visit"),
    baseline_fallback={
        "CREA": "baseline_CREA",
        "CystatinC": "baseline_CystatinC",
        "ALB": "baseline_ALB",
        "log_PRO24H": "baseline_log_PRO24H",
        "UA": "baseline_UA",
        "age_at_visit": "age_at_biopsy",
    },
)

_FULL_COMORBIDITY_ANALYSIS = ClinicalVariableSet(
    name="full_comorbidity_analysis",
    cox_static_cols=(
        "age_at_biopsy", "gender",
        "baseline_hypertension_analysis", "baseline_diabetes_analysis",
    ),
    dynamic_cols=("CREA", "CystatinC", "ALB", "log_PRO24H", "UA"),
    baseline_fallback={
        "CREA": "baseline_CREA",
        "CystatinC": "baseline_CystatinC",
        "ALB": "baseline_ALB",
        "log_PRO24H": "baseline_log_PRO24H",
        "UA": "baseline_UA",
    },
)

_SETS = {item.name: item for item in (_MINIMAL_CORE, _ROUTINE9, _FULL_COMORBIDITY_ANALYSIS)}


def get_clinical_variable_set(name: str) -> ClinicalVariableSet:
    try:
        return _SETS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(_SETS))
        raise ValueError(f"Unknown clinical variable set {name!r}; choose one of: {choices}") from exc


def clinical_variable_set_names() -> tuple[str, ...]:
    return tuple(sorted(_SETS))
