# DDHIgAN model card

## Summary

DDHIgAN is a biopsy-centered Dynamic-DeepHit model for future kidney failure
(ESKD) risk in IgA nephropathy. The clinical model uses age at biopsy, sex,
creatinine, cystatin C, albumin, 24-hour urine protein, and their available
longitudinal trajectories.

## Evidence and intended use

The model was internally validated in a retrospective single-center cohort of
9,948 patients with biopsy-confirmed IgA nephropathy. It is intended for
research demonstration and methods evaluation only.

It has not undergone external validation, prospective evaluation,
transportability assessment, subgroup safety assessment, decision-curve
analysis, or clinical-utility testing. Outputs are not treatment
recommendations, validated thresholds, or medical advice.

## Input and output contract

- The first longitudinal row is the biopsy-time anchor (`t = 0`).
- Every observation must be at or before the query time; future leakage is
  rejected.
- The public schema accepts 1–256 visits and encodes the most recent 30 using
  the true interval at the truncation boundary.
- The query time equals the latest actual visit.
- The API returns a monotonic future 1–10-year kidney-failure risk curve and a
  pointwise 95% bootstrap prediction interval.
- Query times beyond five years after biopsy are explicitly labelled as
  extrapolation beyond the internal-validation range.

The public request schema prohibits names, record numbers, dates, and other
patient identifiers.

## Distribution

This repository publishes architecture, API and web-client source together with
the validated inference weights and aggregate normalization, calibration, and
predictive-uncertainty artifacts. The bundle contains no optimizer state,
training rows, patient-level predictions, identifiers, or source clinical data.
