# DDHIgAN

Source and inference weights for the DDHIgAN research model and identifier-free
Shiny application: https://lewb-ddhigan.share.connect.posit.cloud/

## Current release

`physv15-rankzero-selectable-20260912` provides an explicit choice of **DDHIgAN**
or **DDHIgAN-CysC-free**. Both use the confirmed 9,947-patient source, 11-year
internal support, seed 316, and likelihood/ranking/longitudinal weights 1/0/1.
CysC-free omits biopsy and longitudinal cystatin C, its missingness channel,
and its auxiliary prediction output. Select it explicitly; it is not an
automatic fallback.

Deployment averages predictions from the five saved folds, each using its own
normalization. Internal validation used the held-out fold for each patient;
its reported performance does not directly validate this deployment average.
No additional recalibration or individual confidence interval is supplied.
The earlier release remains available in Git history.

## Manuscript analysis code

The model-training, preprocessing, seven-model evaluation, and independent
patient-level refitting source is available in [`analysis/`](analysis/README.md).
That source release includes scientific configurations, dependency information,
synthetic verification tests, and file hashes. Clinical data and private training
artifacts require access through the corresponding authors for collaborative
research; they are not included in the repository.

## Run the API

```bash
python -m pip install -r requirements-api.txt
# Set a private DDHIGAN_API_KEY through the environment.
export DDHIGAN_BUNDLE_DIR=model-bundle
uvicorn python.deployment.ddhigan_api.app:app --host 127.0.0.1 --port 8091
```

Send `model_id` as `DDHIgAN` (default) or `DDHIgAN-CysC-free` to
`POST /ddhigan/v1/predict`, with the existing `X-API-Key` authentication.
CysC-free requests must omit cystatin C or set it to null.
`GET /v1/model-info` lists available models; `/health/ready` checks readiness.

The contract accepts 1–256 identifier-free, date-free visits, starting at
biopsy (`t=0`), with strictly increasing times. Query time is the last real
visit. The latest 30 records are encoded with the training encoder's zero first interval after truncation. Proteinuria uses the natural log for positive values; zero is
missing. Output is future 1–10-year ESKD risk. Queries after year 5 are labelled
as extrapolation beyond the internal-validation range.

## Web client and verification

Run `shiny::runApp("web")` with server-side `DDHIGAN_API_URL` and
`DDHIGAN_API_KEY`. Never store secrets in source or dependency manifests.

```bash
(cd model-bundle && sha256sum -c SHA256SUMS)
DDHIGAN_BUNDLE_DIR=model-bundle python -m unittest tests.test_latest_selectable_api -v
```

The bundle contains only inference tensors and aggregate normalization and
provenance metadata. Training rows, identifiers, patient predictions,
optimizer state, clinical text, and credentials are excluded.

Source: Apache-2.0. Weights: CC BY-NC 4.0, noncommercial research use.
This retrospective single-center research demonstration is not a medical
device, clinical decision rule, or substitute for clinical judgment.
