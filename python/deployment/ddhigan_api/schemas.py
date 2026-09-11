from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Biomarkers(BaseModel):
    model_config = ConfigDict(extra="forbid")

    creatinine_mg_dl: float | None = Field(default=None, ge=0)
    cystatin_c_mg_l: float | None = Field(default=None, ge=0)
    albumin_g_l: float | None = Field(default=None, ge=0)
    proteinuria_g_24h: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def finite_values(self):
        for name in (
            "creatinine_mg_dl", "cystatin_c_mg_l", "albumin_g_l",
            "proteinuria_g_24h",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class StaticInputs(Biomarkers):
    age_at_biopsy_years: float = Field(ge=0, le=120)
    sex: Literal["female", "male"]


class Visit(Biomarkers):
    time_years: float = Field(ge=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_id: Literal["DDHIgAN", "DDHIgAN-CysC-free"] = "DDHIgAN"
    schema_version: Literal["1.0"]
    query_time_years: float = Field(ge=0)
    kidney_failure_free_at_query: Literal[True]
    static: StaticInputs
    visits: list[Visit] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_timeline(self):
        times = [visit.time_years for visit in self.visits]
        if not math.isfinite(self.query_time_years):
            raise ValueError("query_time_years must be finite")
        if not all(math.isfinite(value) for value in times):
            raise ValueError("visit times must be finite")
        if not math.isclose(times[0], 0.0, abs_tol=1e-8):
            raise ValueError("the first visit must be the t=0 biopsy anchor")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("visit times must be strictly increasing")
        if any(value > self.query_time_years + 1e-8 for value in times):
            raise ValueError("future visits are prohibited")
        if not math.isclose(times[-1], self.query_time_years, abs_tol=1e-8):
            raise ValueError("query time must equal the last actual visit")
        if self.model_id == "DDHIgAN-CysC-free":
            if self.static.cystatin_c_mg_l is not None or any(v.cystatin_c_mg_l is not None for v in self.visits):
                raise ValueError("CysC-free requests must omit cystatin C")
        last = self.visits[-1]
        if all(value is None for name, value in last if name != "time_years"):
            raise ValueError("the query visit must contain at least one biomarker")
        return self


class FirstVisitInputs(Biomarkers):
    age_at_t0_years: float = Field(ge=0, le=120)
    sex: Literal["female", "male"]


class FirstVisitVisit(Biomarkers):
    time_years: float = Field(ge=0)


class PredictionRequestV2(BaseModel):
    """Date-free request contract for the isolated first-visit model."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"]
    biopsy_confirmed: Literal[True]
    history_complete_since_t0: Literal[True]
    kidney_failure_free_at_query: Literal[True]
    query_time_years: float = Field(ge=0)
    static: FirstVisitInputs
    visits: list[FirstVisitVisit] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def validate_timeline(self):
        if not math.isfinite(self.query_time_years):
            raise ValueError("query_time_years must be finite")
        times = [visit.time_years for visit in self.visits]
        if not all(math.isfinite(value) for value in times):
            raise ValueError("visit times must be finite")
        if not math.isclose(times[0], 0.0, abs_tol=1e-8):
            raise ValueError("the first visit must be t0")
        if any(right <= left for left, right in zip(times, times[1:])):
            raise ValueError("visit times must be strictly increasing")
        if any(value > self.query_time_years + 1e-8 for value in times):
            raise ValueError("future visits are prohibited")
        if not math.isclose(times[-1], self.query_time_years, abs_tol=1e-8):
            raise ValueError("query time must equal the last actual visit")
        first = self.visits[0]
        if first.creatinine_mg_dl is None or all(
            value is None for value in (
                first.cystatin_c_mg_l, first.albumin_g_l, first.proteinuria_g_24h,
            )
        ):
            raise ValueError("t0 requires creatinine and at least one other core marker")
        pairs = (
            (self.static.creatinine_mg_dl, first.creatinine_mg_dl),
            (self.static.cystatin_c_mg_l, first.cystatin_c_mg_l),
            (self.static.albumin_g_l, first.albumin_g_l),
            (self.static.proteinuria_g_24h, first.proteinuria_g_24h),
        )
        if any(
            (left is None) != (right is None)
            or (left is not None and not math.isclose(left, right, rel_tol=1e-8, abs_tol=1e-8))
            for left, right in pairs
        ):
            raise ValueError("static t0 biomarkers must equal the first visit")
        return self


class RecentStaticInputs(BaseModel):
    """The only static inputs allowed for the date-free recent models."""

    model_config = ConfigDict(extra="forbid")

    age_at_query_years: float = Field(ge=0, le=120)
    sex: Literal["female", "male"]


class RecentVisit(Biomarkers):
    """One chronologically ordered visit cluster without an absolute date."""

    intervisit_gap_years: float = Field(ge=0, le=5)


class PredictionRequestV3(BaseModel):
    """Date-free schema for DDHIgAN-Recent3y and DDHIgAN-Recent5y.

    The endpoint route selects the 3- or 5-year model.  Thus a request has no
    biopsy date, query time, first-visit offset, patient identifier, or
    absolute calendar date; the final supplied visit is the prediction anchor.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["3.0"]
    kidney_failure_free_at_query: Literal[True]
    static: RecentStaticInputs
    visits: list[RecentVisit] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def validate_recent_timeline(self):
        gaps = [visit.intervisit_gap_years for visit in self.visits]
        if not math.isclose(gaps[0], 0.0, abs_tol=1e-8):
            raise ValueError("the first recent visit must have intervisit_gap_years=0")
        if any(not math.isfinite(value) for value in gaps):
            raise ValueError("intervisit gaps must be finite")
        if any(value <= 0 for value in gaps[1:]):
            raise ValueError("recent visits must be strictly ordered with positive later gaps")
        if sum(gaps) > 5.0 + 1e-8:
            raise ValueError("recent visit history exceeds the maximum 5-year schema support")
        last = self.visits[-1]
        companion_present = any(value is not None for value in (
            last.cystatin_c_mg_l, last.albumin_g_l,
        )) or (last.proteinuria_g_24h is not None and last.proteinuria_g_24h > 0)
        if last.creatinine_mg_dl is None or not companion_present:
            raise ValueError("the query visit requires CREA and at least one other core marker")
        return self
