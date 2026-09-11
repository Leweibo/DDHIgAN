# Thin, identifier-free client for the DDHIgAN research-pilot API.

ddhigan_api_settings <- function() {
  url <- Sys.getenv("DDHIGAN_API_URL", unset = "")
  key <- Sys.getenv("DDHIGAN_API_KEY", unset = "")
  if (!nzchar(url) || !nzchar(key)) {
    stop("DDHIgAN API is not configured; prediction is unavailable.", call. = FALSE)
  }
  ca_bundle <- Sys.getenv("DDHIGAN_CA_BUNDLE", unset = "")
  list(url = sub("/$", "", url), key = key, ca_bundle = ca_bundle)
}

ddhigan_predict <- function(static, visits, query_time_years,
                            model_id = "DDHIgAN",
                            connect_timeout_seconds = 2,
                            response_timeout_seconds = 60) {
  if (!model_id %in% c("DDHIgAN", "DDHIgAN-CysC-free")) stop("Unknown model", call. = FALSE)
  if (identical(model_id, "DDHIgAN-CysC-free")) {
    static$cystatin_c_mg_l <- NULL
    visits$cystatin_c_mg_l <- NULL
  }
  settings <- ddhigan_api_settings()
  allowed_static <- c("age_at_biopsy_years", "sex", "creatinine_mg_dl",
                      "cystatin_c_mg_l", "albumin_g_l", "proteinuria_g_24h")
  allowed_visit <- c("time_years", "creatinine_mg_dl", "cystatin_c_mg_l",
                     "albumin_g_l", "proteinuria_g_24h")
  if (nrow(visits) > 256L || !all(names(static) %in% allowed_static) ||
      !all(names(visits) %in% allowed_visit)) {
    stop("DDHIgAN payload exceeds 256 visits or contains an unapproved field; no prediction was sent.", call. = FALSE)
  }
  visit_rows <- unname(lapply(seq_len(nrow(visits)), function(index) {
    values <- as.list(visits[index, allowed_visit[allowed_visit %in% names(visits)], drop = FALSE])
    lapply(values, function(value) if (length(value) == 0 || is.na(value)) NULL else unname(value))
  }))
  payload <- list(
    schema_version = "1.0",
    model_id = model_id,
    query_time_years = unname(query_time_years),
    kidney_failure_free_at_query = TRUE,
    static = static[allowed_static[allowed_static %in% names(static)]],
    visits = visit_rows
  )
  request <- httr2::request(paste0(settings$url, "/ddhigan/v1/predict")) |>
    httr2::req_headers(`X-API-Key` = settings$key) |>
    httr2::req_body_json(payload, auto_unbox = TRUE) |>
    httr2::req_timeout(response_timeout_seconds) |>
    httr2::req_options(connecttimeout = connect_timeout_seconds) |>
    httr2::req_error(is_error = function(resp) FALSE)
  if (nzchar(settings$ca_bundle)) {
    if (!file.exists(settings$ca_bundle)) {
      stop("DDHIgAN CA certificate is unavailable; prediction has been withheld.", call. = FALSE)
    }
    request <- request |> httr2::req_options(cainfo = settings$ca_bundle)
  }
  response <- tryCatch(httr2::req_perform(request), error = function(error) NULL)
  if (is.null(response)) {
    stop("DDHIgAN API is unavailable; prediction has been withheld.", call. = FALSE)
  }
  status <- httr2::resp_status(response)
  if (status != 200) {
    message <- switch(
      as.character(status),
      `401` = "DDHIgAN API authentication failed.",
      `403` = "DDHIgAN API access was forbidden.",
      `422` = "DDHIgAN input is incomplete or violates the time/unit contract.",
      `429` = "DDHIgAN API rate limit was reached; retry later.",
      if (status >= 500) "DDHIgAN API is unavailable; prediction has been withheld."
      else paste("DDHIgAN API request failed with status", status)
    )
    stop(message, call. = FALSE)
  }
  httr2::resp_body_json(response, simplifyVector = TRUE)
}

ddhigan_normalize_unit <- function(value) {
  tolower(gsub("[[:space:]]", "", as.character(value)))
}

ddhigan_pick_baseline_marker <- function(lab_rows, marker, biopsy_date) {
  rows <- lab_rows[lab_rows$REPORT_ITEM_STANDARD == marker, , drop = FALSE]
  if (!nrow(rows)) return(NA_real_)
  rows$days_from_biopsy <- as.numeric(rows$reportDate - biopsy_date)
  candidates <- rows[rows$days_from_biopsy >= -28 & rows$days_from_biopsy <= 0, , drop = FALSE]
  if (!nrow(candidates)) {
    candidates <- rows[rows$days_from_biopsy >= -60 & rows$days_from_biopsy <= 7, , drop = FALSE]
  }
  if (!nrow(candidates)) return(NA_real_)
  nearest_distance <- min(abs(candidates$days_from_biopsy))
  nearest <- candidates[abs(candidates$days_from_biopsy) == nearest_distance, , drop = FALSE]
  stats::median(nearest$result, na.rm = TRUE)
}

ddhigan_cluster_marker_visits <- function(lab_rows, biopsy_date, eps_days = 15) {
  rows <- unique(lab_rows[, c("reportDate", "REPORT_ITEM_STANDARD", "result"), drop = FALSE])
  rows <- rows[order(rows$reportDate), , drop = FALSE]
  date_gap <- c(NA_real_, diff(as.numeric(rows$reportDate)))
  rows$visit_cluster <- cumsum(seq_len(nrow(rows)) == 1L | (!is.na(date_gap) & date_gap > eps_days))
  rows$date_numeric <- as.numeric(rows$reportDate)

  visit_dates <- stats::aggregate(
    rows$date_numeric,
    by = list(visit_cluster = rows$visit_cluster),
    FUN = function(value) round(stats::median(value, na.rm = TRUE))
  )
  names(visit_dates)[2] <- "date_numeric"
  marker_values <- stats::aggregate(
    rows$result,
    by = list(visit_cluster = rows$visit_cluster, marker = rows$REPORT_ITEM_STANDARD),
    FUN = stats::median, na.rm = TRUE
  )
  names(marker_values)[3] <- "value"
  wide <- reshape(marker_values, idvar = "visit_cluster", timevar = "marker", direction = "wide")
  names(wide) <- sub("^value\\.", "", names(wide))
  wide <- merge(visit_dates, wide, by = "visit_cluster", all.x = TRUE, sort = TRUE)
  wide$reportDate <- as.Date(wide$date_numeric, origin = "1970-01-01")
  for (marker in c("CREA", "CystatinC", "ALB", "PRO24H")) {
    if (!marker %in% names(wide)) wide[[marker]] <- NA_real_
  }
  wide <- wide[order(wide$reportDate), c("reportDate", "CREA", "CystatinC", "ALB", "PRO24H"), drop = FALSE]

  delta_days <- as.numeric(wide$reportDate - biopsy_date)
  anchor_index <- order(abs(delta_days), delta_days < 0, delta_days)[1]
  anchor <- wide[anchor_index, , drop = FALSE]
  after_biopsy <- wide[delta_days >= 0 & seq_len(nrow(wide)) != anchor_index, , drop = FALSE]
  anchor$time_years <- 0
  after_biopsy$time_years <- as.numeric(after_biopsy$reportDate - biopsy_date) / 365.25
  visits <- rbind(anchor, after_biopsy)
  visits <- visits[order(visits$time_years), , drop = FALSE]
  visits <- visits[!duplicated(visits$time_years), , drop = FALSE]
  rownames(visits) <- NULL
  visits
}

ddhigan_prepare_clinical_input <- function(baseinfo, labs, biopsy_date) {
  biopsy_date <- as.Date(biopsy_date)
  if (length(biopsy_date) != 1L || is.na(biopsy_date)) {
    stop("未找到可用的肾活检日期，不能生成 DDHIgAN 预测。", call. = FALSE)
  }
  required_lab_columns <- c("REPORT_ITEM_STANDARD", "result", "reportDate", "units")
  if (!all(required_lab_columns %in% names(labs))) {
    stop("检验数据缺少 DDHIgAN 所需字段。", call. = FALSE)
  }
  if (!nrow(baseinfo) || !all(c("dateOfBirth", "sexValue") %in% names(baseinfo))) {
    stop("患者基础资料不完整，不能生成 DDHIgAN 预测。", call. = FALSE)
  }

  marker_contract <- list(
    CREA = list(field = "creatinine_mg_dl", units = c("mg/dl", "mg/dl.")),
    CystatinC = list(field = "cystatin_c_mg_l", units = c("mg/l")),
    ALB = list(field = "albumin_g_l", units = c("g/l")),
    PRO24H = list(field = "proteinuria_g_24h", units = c("g/24h", "g/24hr", "g/day", "g/24hours"))
  )
  lab_rows <- labs[labs$REPORT_ITEM_STANDARD %in% names(marker_contract), , drop = FALSE]
  lab_rows$reportDate <- as.Date(lab_rows$reportDate)
  lab_rows$result <- suppressWarnings(as.numeric(lab_rows$result))
  lab_rows$normalized_unit <- ddhigan_normalize_unit(lab_rows$units)
  lab_rows <- lab_rows[!is.na(lab_rows$reportDate) & is.finite(lab_rows$result) & lab_rows$result >= 0, , drop = FALSE]

  accepted <- logical(nrow(lab_rows))
  for (index in seq_len(nrow(lab_rows))) {
    contract <- marker_contract[[as.character(lab_rows$REPORT_ITEM_STANDARD[index])]]
    accepted[index] <- lab_rows$normalized_unit[index] %in% contract$units
  }
  rejected <- which(!accepted)
  ignored_units <- character()
  if (length(rejected)) {
    rejected_unit <- trimws(as.character(lab_rows$units[rejected]))
    rejected_unit[is.na(rejected_unit) | !nzchar(rejected_unit)] <- "空白单位"
    ignored_units <- unique(paste0(
      lab_rows$REPORT_ITEM_STANDARD[rejected], " [", rejected_unit, "]"
    ))
  }
  lab_rows <- lab_rows[accepted, , drop = FALSE]
  if (!nrow(lab_rows)) {
    stop("没有单位可验证的 DDHIgAN 纵向检验记录。", call. = FALSE)
  }

  baseline_values <- vapply(
    names(marker_contract),
    function(marker) ddhigan_pick_baseline_marker(lab_rows, marker, biopsy_date),
    numeric(1)
  )
  visits <- ddhigan_cluster_marker_visits(lab_rows, biopsy_date)

  date_of_birth <- as.Date(baseinfo$dateOfBirth[1])
  age <- as.numeric(biopsy_date - date_of_birth) / 365.25
  if (!is.finite(age) || age < 0 || age > 120) {
    stop("活检年龄无法从基础资料中可靠计算。", call. = FALSE)
  }
  sex_value <- trimws(as.character(baseinfo$sexValue[1]))
  sex <- if (sex_value %in% c("女", "female", "Female", "F")) "female" else if (
    sex_value %in% c("男", "male", "Male", "M")
  ) "male" else stop("性别编码无法映射到 DDHIgAN 合同。", call. = FALSE)

  clinical_visits <- data.frame(
    time_years = visits$time_years,
    creatinine_mg_dl = visits$CREA,
    cystatin_c_mg_l = visits$CystatinC,
    albumin_g_l = visits$ALB,
    proteinuria_g_24h = visits$PRO24H,
    check.names = FALSE
  )
  if (nrow(clinical_visits) > 256L) {
    stop("合格的 DDHIgAN 聚类访视超过 API 的 256 条接收上限；未发送预测请求。", call. = FALSE)
  }
  static <- list(
    age_at_biopsy_years = unname(age), sex = sex,
    creatinine_mg_dl = unname(baseline_values[["CREA"]]),
    cystatin_c_mg_l = unname(baseline_values[["CystatinC"]]),
    albumin_g_l = unname(baseline_values[["ALB"]]),
    proteinuria_g_24h = unname(baseline_values[["PRO24H"]])
  )
  static <- lapply(static, function(value) if (length(value) == 0L || is.na(value)) NULL else value)
  warnings <- character()
  if (length(ignored_units)) warnings <- c(warnings, paste0("因单位不符合合同而忽略：", paste(ignored_units, collapse = "、")))
  list(
    static = static,
    visits = clinical_visits,
    query_time_years = unname(tail(clinical_visits$time_years, 1)),
    biopsy_date = biopsy_date,
    query_date = tail(visits$reportDate, 1),
    local_warnings = warnings
  )
}

ddhigan_disclosure <- function(result, query_time_years) {
  list(
    notice = "研究性临床试点，不替代临床判断",
    model_version = result$model_version,
    query_time_years = query_time_years,
    missing_warnings = result$warnings$missing,
    distribution_warnings = result$warnings$distribution,
    late_followup_extrapolation = result$warnings$late_followup_extrapolation,
    history_processing = result$history_processing,
    calibration_notice = paste0(
      "点预测采用 provisional_pooled_oof 校准；浅红阴影为该患者风险预测的 95% 点态 bootstrap 区间",
      "（1,000 次患者 bootstrap 重拟合），表示重复抽样并重新训练、标准化及 OOB 校准时预测值的变动范围。",
      "该区间未涵盖未测量因素、模型设定偏差及不同应用人群带来的不确定性，也不是外部验证。"
    )
  )
}
