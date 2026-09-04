# Aggregate-only descriptive longitudinal mixed-model support for DDHIgAN plots.

.ddhigan_history_lmm_cache <- new.env(parent = emptyenv())

ddhigan_validate_history_lmm <- function(artifact) {
  expected_markers <- c("CREA", "CystatinC", "PRO24H")
  if (!is.list(artifact) ||
      !identical(artifact$artifact_version, "ddhigan_history_lmm_v1") ||
      !identical(artifact$purpose, "descriptive_history_visualization_only_not_ddhigan_inference") ||
      !identical(names(artifact$markers), expected_markers)) {
    stop("历史混合模型工件版本或结构无效。", call. = FALSE)
  }
  if (!identical(artifact$source$snapshot_version, "physv11_expanded_ge1y_20260813") ||
      !identical(artifact$source$snapshot_status, "READY") ||
      as.integer(artifact$source$patients) != 9948L ||
      as.integer(artifact$source$longitudinal_rows) != 198374L) {
    stop("历史混合模型工件来源校验失败。", call. = FALSE)
  }
  for (name in expected_markers) {
    model <- artifact$markers[[name]]
    covariance <- model$random_effect_covariance
    valid <- is.list(model) && length(model$fixed_effects) == 4L &&
      all(is.finite(model$fixed_effects)) && identical(dim(covariance), c(4L, 4L)) &&
      all(is.finite(covariance)) && max(abs(covariance - t(covariance))) <= 1e-8 &&
      max(abs(covariance - diag(diag(covariance)))) <= 1e-8 &&
      min(eigen(covariance, symmetric = TRUE, only.values = TRUE)$values) > 0 &&
      length(model$spline$boundary_knots) == 2L &&
      all(is.finite(c(model$spline$knots, model$spline$boundary_knots))) &&
      length(model$training_time_range) == 2L &&
      all(is.finite(model$training_time_range)) && diff(model$training_time_range) > 0 &&
      is.finite(model$residual_variance) && model$residual_variance > 0 &&
      model$transform %in% c("identity", "log_positive")
    if (!valid) stop(paste0("历史混合模型参数无效：", name), call. = FALSE)
  }
  invisible(artifact)
}

ddhigan_load_history_lmm <- function(path = Sys.getenv("DDHIGAN_HISTORY_LMM_PATH", unset = "")) {
  if (!nzchar(path) || !file.exists(path)) {
    stop("历史混合模型工件不可用；风险摘要与风险曲线不受影响。", call. = FALSE)
  }
  info <- file.info(path)
  key <- paste(normalizePath(path, winslash = "/", mustWork = TRUE), info$size,
               as.numeric(info$mtime), sep = "|")
  if (!identical(.ddhigan_history_lmm_cache$key, key)) {
    artifact <- tryCatch(readRDS(path), error = function(error) NULL)
    if (is.null(artifact)) stop("历史混合模型工件无法读取；风险摘要与风险曲线不受影响。", call. = FALSE)
    ddhigan_validate_history_lmm(artifact)
    .ddhigan_history_lmm_cache$key <- key
    .ddhigan_history_lmm_cache$artifact <- artifact
  }
  .ddhigan_history_lmm_cache$artifact
}

ddhigan_history_ns_basis <- function(time, model) {
  basis <- splines::ns(
    time,
    knots = model$spline$knots,
    Boundary.knots = model$spline$boundary_knots,
    intercept = FALSE
  )
  unname(cbind(1, basis))
}

ddhigan_history_eb_curve <- function(times, values, encoded, model, grid_points = 160L) {
  display_valid <- is.finite(times) & is.finite(values) & encoded
  model_valid <- display_valid
  response <- values
  if (identical(model$transform, "log_positive")) {
    model_valid <- model_valid & values > 0
    response <- rep(NA_real_, length(values))
    response[model_valid] <- log(values[model_valid])
  }
  if (length(unique(times[model_valid])) < 2L) {
    return(list(curve = data.frame(time = numeric(), value = numeric()),
                conditioning_n = sum(model_valid), method = "observed"))
  }
  lower <- max(min(times[display_valid]), model$training_time_range[[1L]])
  upper <- min(max(times[display_valid]), model$training_time_range[[2L]])
  if (!is.finite(lower) || !is.finite(upper) || upper <= lower) {
    return(list(curve = data.frame(time = numeric(), value = numeric()),
                conditioning_n = sum(model_valid), method = "observed"))
  }

  z <- ddhigan_history_ns_basis(times[model_valid], model)
  beta <- as.numeric(model$fixed_effects)
  covariance <- model$random_effect_covariance
  residual <- response[model_valid] - as.vector(z %*% beta)
  marginal <- z %*% covariance %*% t(z) +
    diag(model$residual_variance, nrow(z))
  random_effect <- tryCatch(
    as.vector(covariance %*% t(z) %*% solve(marginal, residual)),
    error = function(error) rep(NA_real_, 4L)
  )
  if (any(!is.finite(random_effect))) {
    return(list(curve = data.frame(time = numeric(), value = numeric()),
                conditioning_n = sum(model_valid), method = "observed"))
  }
  grid <- seq(lower, upper, length.out = as.integer(grid_points))
  prediction <- as.vector(ddhigan_history_ns_basis(grid, model) %*% (beta + random_effect))
  if (identical(model$transform, "log_positive")) prediction <- exp(prediction)
  prediction <- pmax(0, prediction)
  keep <- is.finite(prediction)
  if (sum(keep) < 2L) {
    return(list(curve = data.frame(time = numeric(), value = numeric()),
                conditioning_n = sum(model_valid), method = "observed"))
  }
  list(curve = data.frame(time = grid[keep], value = prediction[keep]),
       conditioning_n = sum(model_valid), method = "lmm")
}
