# R/utils.R
# Shared helper functions for IgAN analysis pipeline

calc_egfr <- function(crea, age, sex_female) {
  # CKD-EPI 2021 (race-free)
  # crea in mg/dL, age in years, sex_female: TRUE/FALSE
  k <- ifelse(sex_female, 0.7, 0.9)
  alpha <- ifelse(sex_female, -0.241, -0.302)
  egfr <- 142 * (pmin(crea / k, 1) ^ alpha) * (pmax(crea / k, 1) ^ -1.2) * (0.9938 ^ age)
  egfr <- ifelse(sex_female, egfr * 1.012, egfr)
  return(egfr)
}

make_time_grid <- function() {
  c(0, 0.5, 1, 2, 3, 5, 7, 10, 15)
}

parse_urine_glucose_dipstick_project <- function(result_text, result_num = NULL) {
  n <- length(result_text)
  if (n == 0) return(numeric())

  text <- as.character(result_text)
  text[is.na(result_text)] <- NA_character_
  text <- trimws(text)
  text <- gsub("（", "(", text, fixed = TRUE)
  text <- gsub("）", ")", text, fixed = TRUE)
  text <- gsub("[[:space:]]+", "", text)

  if (is.null(result_num)) {
    numeric_value <- rep(NA_real_, n)
  } else {
    numeric_value <- suppressWarnings(as.numeric(result_num))
    if (length(numeric_value) == 1 && n != 1) {
      numeric_value <- rep(numeric_value, n)
    }
  }

  out <- rep(NA_real_, n)
  blank <- is.na(text) | !nzchar(text)
  text_nonblank <- ifelse(blank, "", text)

  out[grepl("阴性|正常|negative", text_nonblank, ignore.case = TRUE) |
        text_nonblank %in% c("-", "0", "0.0")] <- 0
  out[grepl("弱阳性|±|\\+-", text_nonblank)] <- 0.5

  # Parse explicit dipstick grades before numeric concentration labels.
  out[is.na(out) & grepl("4\\+|\\+\\+\\+\\+", text_nonblank)] <- 4
  out[is.na(out) & grepl("3\\+|\\+\\+\\+", text_nonblank)] <- 3
  out[is.na(out) & grepl("2\\+|\\+\\+", text_nonblank)] <- 2
  out[is.na(out) & grepl("1\\+|\\+", text_nonblank)] <- 1

  numeric_from_text <- suppressWarnings(as.numeric(gsub("[^0-9.]+", "", text_nonblank)))
  numeric_from_text[!is.na(text) & !grepl("[0-9]", text_nonblank)] <- NA_real_
  numeric_label <- ifelse(is.na(numeric_from_text), numeric_value, numeric_from_text)

  out[is.na(out) & !is.na(numeric_label) & numeric_label == 50] <- 0
  out[is.na(out) & !is.na(numeric_label) & numeric_label == 100] <- 1
  out[is.na(out) & !is.na(numeric_label) & numeric_label == 250] <- 2
  out[is.na(out) & !is.na(numeric_label) & numeric_label == 500] <- 3
  out[is.na(out) & !is.na(numeric_label) & numeric_label == 2000] <- 4

  out
}

fast_followup_cluster_long <- function(
  marker_rows,
  eps_days = 15,
  id_col = "patientId",
  date_col = "reportDate",
  item_col = "marker",
  value_col = "value",
  aggregators = list(),
  default_numeric = "median"
) {
  required <- c(id_col, date_col, item_col, value_col)
  missing <- setdiff(required, names(marker_rows))
  if (length(missing) > 0) {
    stop("Missing marker row column(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }
  if (!requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("tidyr", quietly = TRUE)) {
    stop("fast_followup_cluster_long requires dplyr and tidyr.", call. = FALSE)
  }
  `%>%` <- dplyr::`%>%`

  aggregate_one <- function(v, key) {
    fn <- aggregators[[key]]
    if (!is.null(fn)) return(fn(v))
    if (is.numeric(v)) {
      if (all(is.na(v))) return(NA_real_)
      if (identical(default_numeric, "mean")) return(mean(v, na.rm = TRUE))
      return(stats::median(v, na.rm = TRUE))
    }
    v <- v[!is.na(v)]
    if (length(v) == 0) NA else v[[1]]
  }

  rows <- marker_rows %>%
    dplyr::transmute(
      .followup_id = as.character(.data[[id_col]]),
      .followup_date = as.Date(.data[[date_col]]),
      .followup_item = as.character(.data[[item_col]]),
      .followup_value = .data[[value_col]]
    ) %>%
    dplyr::filter(
      !is.na(.followup_id), .followup_id != "",
      !is.na(.followup_date),
      !is.na(.followup_item), .followup_item != "",
      !is.na(.followup_value)
    ) %>%
    dplyr::distinct() %>%
    dplyr::arrange(.followup_id, .followup_date)

  if (nrow(rows) == 0) return(rows)

  clustered <- rows %>%
    dplyr::group_by(.followup_id) %>%
    dplyr::mutate(
      .date_gap = as.numeric(.followup_date - dplyr::lag(.followup_date)),
      visit_cluster = cumsum(dplyr::row_number() == 1L | .date_gap > eps_days)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::select(-.date_gap)

  visit_counts <- clustered %>%
    dplyr::distinct(.followup_id, visit_cluster) %>%
    dplyr::count(.followup_id, name = "visit_count")

  visit_meta <- clustered %>%
    dplyr::group_by(.followup_id, visit_cluster) %>%
    dplyr::summarise(
      visit_date = as.Date(round(stats::median(as.numeric(.followup_date), na.rm = TRUE)), origin = "1970-01-01"),
      visit_start_date = min(.followup_date, na.rm = TRUE),
      visit_end_date = max(.followup_date, na.rm = TRUE),
      visit_records = dplyr::n(),
      visit_distinct_dates = dplyr::n_distinct(.followup_date),
      .groups = "drop"
    ) %>%
    dplyr::rename(!!id_col := .followup_id)

  values <- clustered %>%
    dplyr::group_by(.followup_id, visit_cluster, .followup_item) %>%
    dplyr::summarise(
      .followup_value = aggregate_one(.followup_value, dplyr::first(.followup_item)),
      .groups = "drop"
    ) %>%
    dplyr::rename(!!id_col := .followup_id) %>%
    tidyr::pivot_wider(names_from = .followup_item, values_from = .followup_value)

  visit_meta %>%
    dplyr::left_join(values, by = c(id_col, "visit_cluster")) %>%
    dplyr::left_join(visit_counts, by = stats::setNames(".followup_id", id_col)) %>%
    dplyr::arrange(.data[[id_col]], visit_date, visit_cluster)
}

recode_rechecked_outcomes <- function(outcome_df) {
  required_cols <- c(
    "ESKD_status", "ESKD_time", "Drop50_status", "Drop50_time",
    "ESKD_status2", "ESKD_time2", "Drop50_status2", "Drop50_time2"
  )
  missing <- setdiff(required_cols, names(outcome_df))
  if (length(missing) > 0) {
    stop("Missing FileMaker outcome column(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }

  outcome_df %>%
    dplyr::mutate(
      ESKD_status = dplyr::if_else(is.na(ESKD_status2), ESKD_status, ESKD_status2),
      ESKD_time = dplyr::if_else(is.na(ESKD_time2), ESKD_time, ESKD_time2),
      Drop50_status = dplyr::if_else(is.na(Drop50_status2), Drop50_status, Drop50_status2),
      Drop50_time = dplyr::if_else(is.na(Drop50_time2), Drop50_time, Drop50_time2)
    ) %>%
    dplyr::select(-ESKD_status2, -ESKD_time2, -Drop50_status2, -Drop50_time2)
}

add_survival_endpoint <- function(df, outcome = c("ESKD", "Drop50")) {
  outcome <- match.arg(outcome)
  prefix <- if (outcome == "ESKD") "eskd" else "drop50"
  time_col <- paste0(prefix, "_time")
  status_col <- paste0(prefix, "_status")

  missing <- setdiff(c(time_col, status_col), names(df))
  if (length(missing) > 0) {
    stop("Missing processed outcome column(s): ", paste(missing, collapse = ", "), call. = FALSE)
  }

  df %>%
    dplyr::mutate(
      event_time = suppressWarnings(as.numeric(.data[[time_col]])),
      event_status = dplyr::case_when(
        as.character(.data[[status_col]]) == "1" ~ 1L,
        as.character(.data[[status_col]]) == "0" ~ 0L,
        TRUE ~ NA_integer_
      )
    )
}

is_isomorphic_urbc_type <- function(urbc_type) {
  text <- trimws(as.character(urbc_type))
  text[is.na(urbc_type)] <- NA_character_
  has_uniform <- grepl("均一", text)
  has_glomerular_hint <- grepl("非均一|不均一|多形|畸形|混合", text)
  ifelse(is.na(text) | !nzchar(text), FALSE, has_uniform & !has_glomerular_hint)
}

prepare_urbc_records <- function(
  df,
  result_col = "result_num",
  unit_transfer_col = "UNIT_Transfer",
  type_col = "URBC_TYPE"
) {
  if (!result_col %in% names(df)) {
    stop("Missing URBC result column: ", result_col, call. = FALSE)
  }

  result_value <- suppressWarnings(as.numeric(df[[result_col]]))
  unit_transfer <- if (unit_transfer_col %in% names(df)) {
    suppressWarnings(as.numeric(df[[unit_transfer_col]]))
  } else {
    rep(1, length(result_value))
  }
  unit_transfer[is.na(unit_transfer)] <- 1

  urbc_type <- if (type_col %in% names(df)) {
    as.character(df[[type_col]])
  } else {
    rep(NA_character_, length(result_value))
  }
  isomorphic <- is_isomorphic_urbc_type(urbc_type)
  urbc <- result_value * unit_transfer
  urbc[is.na(result_value) | result_value < 0 | isomorphic] <- NA_real_

  df$URBC <- urbc
  df$log_URBC <- log(urbc + 1)
  df$URBC_isomorphic_flag <- isomorphic
  df$URBC_type_missing <- is.na(urbc_type) | !nzchar(trimws(urbc_type))
  df
}

write_splits <- function(patient_ids,
                         n_folds = 5,
                         out_dir = file.path("Data", "splits"),
                         seed = 316) {
  if (exists("resolve_project_path", mode = "function")) {
    out_dir <- resolve_project_path(out_dir)
  }
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  set.seed(seed)
  shuffled <- sample(patient_ids)
  folds <- cut(seq_along(shuffled), breaks = n_folds, labels = FALSE)

  for (k in 0:(n_folds - 1)) {
    test_ids <- shuffled[folds == (k + 1)]
    train_ids <- shuffled[folds != (k + 1)]
    write.csv(data.frame(patient_id = train_ids), file.path(out_dir, sprintf("fold_%d_train.csv", k)), row.names = FALSE)
    write.csv(data.frame(patient_id = test_ids), file.path(out_dir, sprintf("fold_%d_test.csv", k)), row.names = FALSE)
  }
}
