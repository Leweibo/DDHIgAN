#!/usr/bin/env Rscript
if (nzchar(Sys.getenv("PCCOX_R_LIBRARY"))) {
  .libPaths(c(Sys.getenv("PCCOX_R_LIBRARY"), .libPaths()))
}

suppressPackageStartupMessages({
  library(partlyconditional)
  library(splines)
})

args <- commandArgs(trailingOnly = TRUE)
arg <- function(name) {
  hit <- which(args == name)
  if (length(hit) != 1L || hit == length(args)) stop("missing argument: ", name)
  args[[hit + 1L]]
}
arg_optional <- function(name, default) {
  hit <- which(args == name)
  if (!length(hit)) return(default)
  if (length(hit) != 1L || hit == length(args)) stop("invalid argument: ", name)
  args[[hit + 1L]]
}
input_dir <- arg("--input-dir")
output_dir <- arg("--output-dir")
fold <- as.integer(arg("--fold"))
landmarks <- as.numeric(strsplit(arg_optional("--landmarks", "0,1,2,3,4,5"), ",", fixed = TRUE)[[1L]])
if (!length(landmarks) || any(!is.finite(landmarks)) || anyDuplicated(landmarks)) stop("invalid landmarks")
prefix <- "pccox_minimal_core"
times <- 1:10
features <- c("age_at_biopsy", "gender", "CREA", "CystatinC", "ALB", "log_PRO24H")

if (dir.exists(output_dir)) stop("refusing to overwrite PCCox output directory")
dir.create(output_dir, recursive = TRUE)
fit_data <- read.csv(file.path(input_dir, "fit.csv"), check.names = FALSE,
                     colClasses = c(patient_id = "character"))
required <- c("patient_id", "stime", "status", "measurement_time", features)
stopifnot(all(required %in% names(fit_data)))
stopifnot(all(fit_data$measurement_time >= 0 & fit_data$measurement_time <= 5))
stopifnot(all(fit_data$measurement_time < fit_data$stime))
stopifnot(!("patient_weight" %in% names(fit_data)))

model <- PC.Cox(
  id = "patient_id", stime = "stime", status = "status",
  measurement.time = "measurement_time", predictors = features,
  data = fit_data,
  additional.formula.pars = "splines::ns(measurement_time, df = 3)"
)
stopifnot(inherits(model, "PC_cox"))
stopifnot(!is.null(model$model.fit$naive.var))
stopifnot(grepl("cluster = patient_id", paste(deparse(model$model.fit$call), collapse = ""), fixed = TRUE))
saveRDS(model, file.path(output_dir, sprintf("%s_ESKD_fold%d_model.rds", prefix, fold)))

predict_frame <- function(frame) {
  frame$stime <- frame$duration + frame$measurement_time
  frame$status <- frame$event
  predicted <- predict(model, newdata = frame, prediction.time = times)
  risk_cols <- paste0("risk_", times)
  stopifnot(nrow(predicted) == nrow(frame), all(risk_cols %in% names(predicted)))
  risk <- as.matrix(predicted[, risk_cols, drop = FALSE])
  stopifnot(all(is.finite(risk)), all(risk >= 0 & risk <= 1))
  if (any(apply(risk, 1, function(x) any(diff(x) < -1e-8)))) stop("non-monotone PCCox risk curve")
  out <- data.frame(
    patient_id = as.character(frame$patient_id),
    true_time = frame$duration + frame$measurement_time,
    residual_time = frame$duration,
    true_event = frame$event,
    Tstart = frame$measurement_time,
    check.names = FALSE
  )
  for (j in seq_along(times)) out[[sprintf("resid_surv_%.1fy", times[[j]])]] <- 1 - risk[, j]
  out$cond_surv_3y <- out[["resid_surv_3.0y"]]
  out$cond_surv_5y <- out[["resid_surv_5.0y"]]
  out$cond_surv_10y <- out[["resid_surv_10.0y"]]
  out$risk_score <- 1 - out$cond_surv_10y
  out
}

for (landmark in landmarks) {
  test <- read.csv(file.path(input_dir, sprintf("test_landmark%d.csv", landmark)),
                   check.names = FALSE, colClasses = c(patient_id = "character"))
  stopifnot(all(test$measurement_time == landmark))
  prediction <- predict_frame(test)
  write.csv(prediction, file.path(
    output_dir, sprintf("%s_ESKD_fold%d_test_landmark%d.csv", prefix, fold, landmark)
  ), row.names = FALSE)
}

provenance <- list(
  status = "COMPLETE", model_key = "pccox", reader_name = "Partly Conditional Cox (PCCox)",
  artifact_prefix = prefix, fold = fold, fit_count = 1L,
  package = as.character(packageVersion("partlyconditional")),
  R = R.version.string, survival = as.character(packageVersion("survival")),
  spline = "splines::ns(measurement_time, df = 3)", cluster_robust = TRUE,
  uses_BLUP = FALSE, uses_patient_visit_weights = FALSE, uses_penalizer = FALSE,
  evaluation_landmarks = landmarks
)
jsonlite::write_json(provenance, file.path(output_dir, sprintf("fold%d_provenance.json", fold)),
                     pretty = TRUE, auto_unbox = TRUE)
cat(sprintf("PCCOX_FOLD_COMPLETE fold=%d rows=%d patients=%d\n", fold, nrow(fit_data), length(unique(fit_data$patient_id))))
