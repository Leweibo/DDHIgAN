# DDHIgAN

Source release for the DDHIgAN Dynamic-DeepHit research model, its protected
FastAPI inference service, and the identifier-free Posit Connect Cloud Shiny
client.

Live research demonstration:
https://lewb-ddhigan.share.connect.posit.cloud/

DDHIgAN estimates future kidney failure (ESKD) risk in biopsy-confirmed IgA
nephropathy from biopsy-time minimal-core clinical variables and longitudinal
marker history. It is a retrospective research model, not a medical device or
a substitute for clinical judgment.

## Repository contents

- `python/deephit/model.py`: Dynamic-DeepHit architecture.
- `python/deployment/ddhigan_api/`: authenticated, metadata-only-logging v1
  inference API and input validation.
- `python/utils/time_grid.py`: half-year discrete-time grid.
- `web/`: bilingual R Shiny client used by the public Connect Cloud app.
- `tests/`: identifier-free API and history-window contract tests.

The validated production inference bundle is included under `model-bundle/`.
Its two `.pt` files are stored with Git LFS. The bundle contains model weights
and aggregate normalization, calibration, uncertainty, contract, provenance,
and checksum metadata. It contains no training rows, patient data, clinical
text, embeddings, identifiers, or patient-level predictions.

Credentials, private deployment configuration, the parent research project,
and raw or processed cohort data are intentionally not included.

## Web client

The Shiny client accepts no patient names, identifiers, or dates. Configure its
backend only through server-side environment variables:

```text
DDHIGAN_API_URL=https://your-api.example/
DDHIGAN_API_KEY=replace-with-a-server-side-secret
```

Never place the values in source files or `manifest.json`. From `web/`, run:

```r
shiny::runApp()
```

For Posit Connect Cloud, publish `web/manifest.json` and configure the two
variables as Secret Variables.

## API development

```bash
python -m pip install -r requirements-api.txt
python -m unittest tests.test_ddhigan_api_allhistory_v1 -v
```

At runtime, provide `DDHIGAN_BUNDLE_DIR=model-bundle` and `DDHIGAN_API_KEY`
through server-side configuration or a secret manager. Verify the bundle first:

```bash
(cd model-bundle && sha256sum -c SHA256SUMS)
```

The input contract permits 1–256 biopsy-anchored visits, encodes the most recent
30, rejects future observations and identifiers, and labels query times beyond
five years as research extrapolation.

## License

Source code is available under the Apache License 2.0. Model weights are
separately licensed for noncommercial research use under CC BY-NC 4.0; see
`WEIGHTS_LICENSE.md`. No clinical-data license is granted because no clinical
data is included.
