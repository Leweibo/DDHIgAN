# Current DDHIgAN model card

Release: physv15-rankzero-selectable-20260912.

Two separately trained predictor sets are available: DDHIgAN and
DDHIgAN-CysC-free. The latter removes all cystatin C input and auxiliary-output
channels. Both use 11-year internal support and 1/0/1 likelihood/ranking/
longitudinal loss weights. The confirmed source cohort contains 9,947 patients.

Predictions average five saved fold models with fold-specific standardization.
No full-cohort refit, additional recalibration, or bootstrap uncertainty fit
was performed for this deployment. Individual confidence intervals are not
available. Fold variation must not be interpreted as such an interval.
Internal validation used held-out-fold predictions, not this deployment mean.

Mean internally validated AUC10 was 0.902 for DDHIgAN and 0.899 for CysC-free.
CysC-free had slightly lower mean discrimination; IBS and calibration
differences were uncertain. These findings support further evaluation as a
potential alternative where cystatin C is unavailable, not equivalence or
clinical replacement. Neither version has external or prospective validation.

Inputs: age, sex, biopsy-time and longitudinal creatinine, albumin and
positive log-proteinuria; the core also uses cystatin C. No identifiers or dates.
Query is the latest actual visit; histories start at biopsy, and the latest
30 of at most 256 records are encoded. Future records are rejected. Query
times greater than five years are marked as extrapolation. Output: monotonic
future 1–10-year ESKD risk. No treatment advice or validated risk threshold.

See model-bundle/latest_manifest.json for tensor and scaler lineage, original
checkpoint hashes, and fixed aggregation. Weights are CC BY-NC 4.0; code is
Apache-2.0. No clinical data are distributed.
