# Manuscript analysis source

This directory provides the model-training, preprocessing, prediction, and
evaluation source for the seven-model DDHIgAN study using the confirmed
9,947-patient cohort. The public web application remains in `../web`, and its
inference API remains in `../python/deployment/ddhigan_api`.

## Scientific entry points

| Purpose | Source |
|---|---|
| DDHIgAN fitting and held-out prediction | `python/deephit/train.py`, `python/deephit/predict.py` |
| Four DDHIgAN predictor sets | `config/physv15_rankzero_standard_20260911/{core,pathology,expanded}.yaml`, `config/physv15_rankzero_sevenmodel_20260911/cysc_free.yaml` |
| Baseline Cox and comparator input preparation | `python/scripts/run_arbitrary_visit_reanalysis_references.py`, `python/scripts/prepare_arbitrary_visit_dynforest.py` |
| PCCox fitting and query-time prediction | `python/scripts/prepare_pccox_inputs.py`, `R/run_arbitrary_visit_pccox.R`, `python/scripts/prepare_pccox_query15.py`, `R/predict_pccox_query15.R`, `R/query_time_survival.R` |
| DynForest fitting and saved-fit prediction | `R/run_arbitrary_visit_dynforest.R`, `R/predict_arbitrary_visit_dynforest_from_rds.R`, `R/dynforest_query_denominator.R` |
| Seven-model evaluation and paired fixed-model bootstrap | `scripts/evaluate_sevenmodel_standard.py`, `scripts/evaluate_rankzero_standard.py`, `scripts/evaluate_query12_recency.py`, `python/evaluation/uno_ibs_cluster_bootstrap.py` |
| Independent 500-replicate patient-level refitting analysis | `scripts/run_fixed_rankzero_bootstrap.py`, `scripts/run_ddhigan_refit_pilot.py`, `python/deephit/fixed_bootstrap.py` |
| Outcome recoding before processed-data generation | `R/utils.R`, function `recode_rechecked_outcomes()` |

The main comparison uses five patient folds, seed 316, seven models, landmarks
0–5 years, a most recent actual record within one year of each query (inclusive),
and future horizons of 1–10 years. DDHIgAN uses 11 years of residual support and
likelihood/ranking/longitudinal loss weights 1/0/1. Landmark queries define
evaluation windows; the main analysis does not fit a new model per landmark.
There are 210 model/fold/landmark prediction cells. The primary fixed-model
bootstrap has 2,000 shared patient draws; the separate refitting analysis has
500 replicates and retains the original outer folds and inner partitions.

Some shared source functions retain diagnostic 3-year metrics or historical
branches. The current manuscript reports the seven named models, primary
10-year AUC, and supporting 5-year AUC, Uno C, IBS, and calibration measures.
Diagnostic fields are not additional manuscript results. Exploratory selection
modules are included only where required by shared source dependencies.

## Data access and reproducibility

Clinical records, patient identifiers, clinical text, pathology embeddings,
fold membership, patient bootstrap draws, patient predictions, training
checkpoints, and private audit/gate files are not distributed in this source
release. Data may be requested from the corresponding authors for collaborative
research, subject to institutional approval and applicable privacy requirements.
See the contact details in the article. Public deployment inference weights
are separately documented at the repository root.

This is a source release, not a stand-alone reproduction of cohort results.
The original orchestration scripts require the approved private data and frozen
intermediate artifacts, provenance manifests, and confirmation files. Historical
host and hash assertions remain in the orchestration source; they are integrity
checks, not portable execution examples. Relative data locations have been
replaced with `data/processed`, `data/splits`, and `data/pathology`; no scientific
setting was changed. Hashes embedded in historical run specifications describe
the original execution artifacts, not the edited public path templates.

`SOURCE_MANIFEST.json` records the SHA-256 of each source file and its public
copy. `SHA256SUMS` covers the complete analysis release. Access to an approved
private reproduction bundle must be arranged before running cohort analyses.

## Environment and source verification

The study used Python 3.10.4, PyTorch 2.0.1, and NumPy 1.26.4 for DDHIgAN.
`requirements-training.txt` specifies that training environment; the comparator
evaluation environment is separately specified in
`config/requirements-seven-model-references.txt`. R scripts require `survival`,
`splines`, `jsonlite`, and `DynForest` or `partlyconditional` (2.0 for PCCox).
`R/utils.R` also uses `dplyr`, `tidyr`, and `lubridate` for data processing.
R fitting scripts record their package versions in their provenance outputs.

Run these checks from this directory; they use synthetic examples and do not
train the cohort models:

```bash
sha256sum -c SHA256SUMS
python -m unittest python.tests.test_query_conditioning \
  python.tests.test_query12_recency python.tests.test_fixed_bootstrap \
  python.tests.test_sevenmodel_cysc_free -v
```

After data access and the original integrity gates are arranged, the trainer
accepts `--config`, `--fold`, and `--outcome ESKD`. `ESKD` is a retained internal
column/CLI name for the study's kidney failure endpoint, not a different outcome.
Use the four configurations listed above. Inspect each script's arguments and
the data contract before executing; source publication itself does not launch
training or bootstrap computation.

Source code is covered by the repository's Apache-2.0 license.
