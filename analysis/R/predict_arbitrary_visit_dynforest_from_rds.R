#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name, default = NULL) {
  hit <- match(name, args)
  if (is.na(hit)) return(default)
  if (hit == length(args)) stop(name, " requires a value")
  args[[hit + 1L]]
}

input_root <- arg("--input-root")
output_dir <- arg("--output-dir")
model_rds <- arg("--model-rds")
rlib <- arg("--rlib")
fold <- as.integer(arg("--fold"))
prediction_cores <- as.integer(arg("--prediction-cores", "4"))
prediction_chunks <- as.integer(arg("--prediction-chunks", "40"))
landmarks <- as.numeric(strsplit(arg("--landmarks"), ",", fixed = TRUE)[[1L]])
if (any(vapply(list(input_root, output_dir, model_rds, rlib), is.null, logical(1))) ||
    !fold %in% 0:4 || !identical(landmarks, c(2, 4))) stop("invalid inference contract")
if (prediction_cores < 1L || prediction_chunks < prediction_cores) stop("invalid prediction concurrency")
if (!file.exists(model_rds)) stop("missing frozen DynForest model")

.libPaths(c(rlib, .libPaths()))
suppressPackageStartupMessages(library(DynForest))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
fold_dir <- file.path(input_root, paste0("fold", fold))
artifact_prefix <- "dynforest_longitudinal_minimal_core_full_outer_train"
read_ids <- function(path) read.csv(path, colClasses = c(patient_id = "character"), check.names = FALSE)
recode_ids <- function(frame, id_map) {
  values <- unname(id_map[frame$patient_id])
  if (anyNA(values)) stop("patient ID missing from DynForest numeric ID map")
  frame$patient_id <- as.integer(values)
  frame
}

train_fixed <- read_ids(file.path(fold_dir, "train_fixed.csv"))
train_gender_levels <- levels(factor(train_fixed$gender))
model <- readRDS(model_rds)

as_patient_time_matrix <- function(prediction, expected_n) {
  values <- as.matrix(prediction$pred_indiv)
  if (nrow(values) == expected_n) return(values)
  if (ncol(values) == expected_n) return(t(values))
  stop("DynForest prediction dimensions do not match the test patients")
}

predict_chunked <- function(time_test, fixed_test, landmark) {
  n_subjects <- nrow(fixed_test)
  nchunks <- max(1L, min(prediction_chunks, n_subjects))
  cores <- max(1L, min(prediction_cores, nchunks))
  chunk_rows <- split(seq_len(n_subjects), cut(seq_len(n_subjects), nchunks, labels = FALSE))
  parts <- parallel::mclapply(chunk_rows, function(rows) {
    prediction <- predict(
      model,
      timeData = time_test[time_test$patient_id %in% fixed_test$patient_id[rows], ],
      fixedData = fixed_test[rows, ], idVar = "patient_id", timeVar = "visit_time",
      t0 = as.numeric(landmark)
    )
    list(risk = as_patient_time_matrix(prediction, length(rows)),
         times = as.numeric(prediction$times))
  }, mc.cores = cores, mc.preschedule = TRUE)
  times <- parts[[1L]]$times
  if (any(!vapply(parts, function(part) identical(part$times, times), logical(1)))) {
    stop("DynForest prediction time grid differs across chunks")
  }
  risk <- do.call(rbind, lapply(parts, function(part) part$risk))
  if (nrow(risk) != n_subjects) stop("DynForest chunked prediction lost subjects")
  list(risk = risk, times = times)
}

written <- character()
for (landmark in landmarks) {
  landmark_dir <- file.path(fold_dir, paste0("landmark", landmark))
  time_test <- read_ids(file.path(landmark_dir, "test_time.csv"))
  fixed_test <- read_ids(file.path(landmark_dir, "test_fixed.csv"))
  truth <- read_ids(file.path(landmark_dir, "truth.csv"))
  if (!identical(as.character(truth$patient_id), as.character(fixed_test$patient_id))) {
    stop("truth and fixed-data patient order mismatch")
  }
  test_id_map <- setNames(seq_len(nrow(fixed_test)), fixed_test$patient_id)
  time_test <- recode_ids(time_test, test_id_map)
  fixed_test <- recode_ids(fixed_test, test_id_map)
  fixed_test$gender <- factor(fixed_test$gender, levels = train_gender_levels)
  predicted <- predict_chunked(time_test, fixed_test, landmark)
  risk <- predicted$risk
  times <- predicted$times
  result <- truth
  for (horizon in 1:10) {
    positions <- which(times <= landmark + horizon)
    if (!length(positions)) stop("DynForest prediction grid does not cover the requested horizon")
    survival <- 1 - risk[, max(positions)]
    result[[sprintf("resid_surv_%.1fy", horizon)]] <- survival
    if (horizon %in% c(3, 5, 10)) result[[paste0("cond_surv_", horizon, "y")]] <- survival
  }
  result$risk_score <- 1 - result$cond_surv_10y
  probabilities <- as.matrix(result[, grep("^(resid_surv_|cond_surv_|risk_score$)", names(result))])
  if (any(!is.finite(probabilities)) || any(probabilities < 0) || any(probabilities > 1)) {
    stop("non-finite or out-of-range DynForest probabilities")
  }
  curve <- as.matrix(result[, sprintf("resid_surv_%.1fy", 1:10)])
  if (any(curve[, -1, drop = FALSE] > curve[, -ncol(curve), drop = FALSE] + 1e-10)) {
    stop("non-monotone DynForest survival curve")
  }
  target <- file.path(output_dir, sprintf(
    "%s_ESKD_fold%d_test_landmark%d.csv", artifact_prefix, fold, landmark
  ))
  if (file.exists(target)) stop("refusing to overwrite DynForest inference output")
  write.csv(result, target, row.names = FALSE, na = "")
  written <- c(written, target)
}

provenance <- list(
  status = "COMPLETE", mode = "inference_only", model_key = "dynforest",
  artifact_prefix = artifact_prefix, fold = fold, fit_count = 0L,
  frozen_model_rds = normalizePath(model_rds),
  package = as.character(packageVersion("DynForest")),
  evaluation_landmarks = landmarks, prediction_cores = prediction_cores,
  prediction_chunks = prediction_chunks, prediction_files = basename(written)
)
jsonlite::write_json(
  provenance,
  file.path(output_dir, sprintf("fold%d_inference_provenance.json", fold)),
  pretty = TRUE, auto_unbox = TRUE
)
cat(sprintf("DYNFOREST_INFERENCE_ONLY_COMPLETE fold=%d predictions=%d\n", fold, length(written)))
