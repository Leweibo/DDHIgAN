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
rlib <- arg("--rlib")
fold <- as.integer(arg("--fold"))
ntree <- as.integer(arg("--ntree", "500"))
ncores <- as.integer(arg("--ncores", "4"))
prediction_cores <- as.integer(arg("--prediction-cores", as.character(min(4L, ncores))))
prediction_chunks <- as.integer(arg("--prediction-chunks", as.character(ncores)))
landmarks <- as.numeric(strsplit(arg("--landmarks", "0,1,2,3,4,5"), ",", fixed = TRUE)[[1L]])
if (any(vapply(list(input_root, output_dir, rlib), is.null, logical(1))) || is.na(fold)) {
  stop("--input-root, --output-dir, --rlib, and --fold are required")
}
if (is.na(ncores) || ncores < 1L || is.na(prediction_cores) || prediction_cores < 1L ||
    prediction_cores > ncores || is.na(prediction_chunks) || prediction_chunks < 1L) {
  stop("invalid training or prediction concurrency")
}
if (!length(landmarks) || any(!is.finite(landmarks)) || anyDuplicated(landmarks)) stop("invalid landmarks")

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

time_train <- read_ids(file.path(fold_dir, "train_time.csv"))
fixed_train <- read_ids(file.path(fold_dir, "train_fixed.csv"))
outcome_train <- read_ids(file.path(fold_dir, "train_outcome.csv"))
train_id_map <- setNames(seq_len(nrow(fixed_train)), fixed_train$patient_id)
time_train <- recode_ids(time_train, train_id_map)
fixed_train <- recode_ids(fixed_train, train_id_map)
outcome_train <- recode_ids(outcome_train, train_id_map)
fixed_train$gender <- factor(fixed_train$gender)
Y <- list(type = "surv", Y = outcome_train[, c("patient_id", "eskd_time", "eskd_status")])
time_models <- list(
  CREA = list(fixed = CREA ~ visit_time, random = ~ visit_time),
  CystatinC = list(fixed = CystatinC ~ visit_time, random = ~ visit_time),
  ALB = list(fixed = ALB ~ visit_time, random = ~ visit_time),
  log_PRO24H = list(fixed = log_PRO24H ~ visit_time, random = ~ visit_time)
)

rds_path <- file.path(
  output_dir, sprintf("%s_ESKD_fold%d_fit.rds", artifact_prefix, fold)
)
if (file.exists(rds_path)) {
  stop("Refusing to reuse or overwrite a formal full-outer-training DynForest fit")
}
model <- do.call(dynforest, list(
  timeData = time_train, fixedData = fixed_train,
  idVar = "patient_id", timeVar = "visit_time", timeVarModel = time_models,
  Y = Y, ntree = ntree, mtry = 3L, nodesize = 5L, minsplit = 10L,
  cause = 1L, ncores = ncores, seed = as.integer(316L + fold), verbose = TRUE
))
saveRDS(model, rds_path)

as_patient_time_matrix <- function(prediction, expected_n) {
  values <- as.matrix(prediction$pred_indiv)
  if (nrow(values) == expected_n) return(values)
  if (ncol(values) == expected_n) return(t(values))
  stop("DynForest prediction dimensions do not match the test patients")
}

predict_chunked <- function(model, time_test, fixed_test, landmark, nchunks, prediction_cores) {
  n_subjects <- nrow(fixed_test)
  nchunks <- max(1L, min(as.integer(nchunks), n_subjects))
  prediction_cores <- max(1L, min(as.integer(prediction_cores), nchunks))
  chunk_rows <- split(seq_len(n_subjects), cut(seq_len(n_subjects), nchunks, labels = FALSE))
  parts <- parallel::mclapply(chunk_rows, function(rows) {
    prediction <- do.call(predict, list(
      object = model,
      timeData = time_test[time_test$patient_id %in% fixed_test$patient_id[rows], ],
      fixedData = fixed_test[rows, ],
      idVar = "patient_id", timeVar = "visit_time", t0 = as.numeric(landmark)
    ))
    list(risk = as_patient_time_matrix(prediction, length(rows)),
         times = as.numeric(prediction$times))
  }, mc.cores = prediction_cores, mc.preschedule = TRUE)
  times <- parts[[1L]]$times
  for (part in parts) {
    if (!identical(part$times, times)) stop("DynForest prediction time grid differs across chunks")
  }
  risk <- do.call(rbind, lapply(parts, function(part) part$risk))
  if (nrow(risk) != n_subjects) stop("DynForest chunked prediction lost subjects")
  list(risk = risk, times = times)
}

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
  fixed_test$gender <- factor(fixed_test$gender, levels = levels(fixed_train$gender))
  predicted <- predict_chunked(
    model, time_test, fixed_test, landmark, prediction_chunks, prediction_cores
  )
  risk <- predicted$risk
  times <- predicted$times
  if (ncol(risk) != length(times)) stop("DynForest prediction time grid mismatch")
  result <- truth
  for (horizon in 1:10) {
    position <- max(which(times <= landmark + horizon))
    if (!is.finite(position)) stop("DynForest prediction grid does not cover the requested horizon")
    survival <- 1 - risk[, position]
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
  write.csv(
    result,
    file.path(output_dir, sprintf(
      "%s_ESKD_fold%d_test_landmark%d.csv",
      artifact_prefix, fold, landmark
    )), row.names = FALSE, na = ""
  )
}

writeLines(c(
  "status=COMPLETE", paste0("fold=", fold), paste0("seed=", 316 + fold),
  paste0("ntree=", ntree), "mtry=3", "nodesize=5", "minsplit=10",
  paste0("training_cores=", ncores), paste0("prediction_chunks=", prediction_chunks),
  paste0("prediction_cores=", prediction_cores), "prediction_mc_preschedule=true",
  "training_reused=false", "fit_population=full_outer_training_fold",
  "training_query_window=0-5", paste0("evaluation_landmarks=", paste(landmarks, collapse = ","))
), file.path(output_dir, sprintf("%s_ESKD_fold%d.env", artifact_prefix, fold)))
