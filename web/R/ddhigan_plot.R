# Identifier-free plotting helpers for the DDHIgAN prediction page.

ddhigan_history_plot_spec <- function(visits, history_processing, query_time_years) {
  required <- c(
    "time_years", "creatinine_mg_dl", "cystatin_c_mg_l",
    "albumin_g_l", "proteinuria_g_24h"
  )
  if (!is.data.frame(visits) || !all(required %in% names(visits))) {
    stop("历史轨迹数据缺少 DDHIgAN 绘图字段。", call. = FALSE)
  }
  count <- function(name) {
    value <- suppressWarnings(as.integer(history_processing[[name]][1]))
    if (!length(value) || is.na(value)) NA_integer_ else value
  }
  received <- count("received_visits")
  encoded <- count("encoded_visits")
  omitted <- count("omitted_visits")
  expected_encoded <- min(nrow(visits), 30L)
  counts_match <- !anyNA(c(received, encoded, omitted)) &&
    received == nrow(visits) && encoded == expected_encoded &&
    omitted == received - encoded
  if (!counts_match) {
    stop(
      "历史访视计数与 API 处理元数据不一致；历史轨迹已停止显示，请核对本次请求。",
      call. = FALSE
    )
  }

  times <- suppressWarnings(as.numeric(visits$time_years))
  query_time_years <- suppressWarnings(as.numeric(query_time_years)[1])
  if (any(!is.finite(times)) || !is.finite(query_time_years)) {
    stop("历史轨迹时间数据不可用。", call. = FALSE)
  }
  encoded_index <- seq_len(nrow(visits)) > omitted
  x_range <- range(c(0, times, query_time_years), finite = TRUE)
  if (diff(x_range) == 0) {
    x_range <- x_range + c(-0.5, 0.5)
  } else {
    span <- diff(x_range)
    x_range <- c(min(0, x_range[1]) - 0.02 * span, x_range[2] + 0.04 * span)
  }

  list(
    visits = visits,
    encoded = encoded_index,
    received_visits = received,
    encoded_visits = encoded,
    omitted_visits = omitted,
    query_time_years = query_time_years,
    xlim = x_range
  )
}

ddhigan_marker_series <- function(history_spec, field, label, color, pch, model) {
  if (!field %in% names(history_spec$visits)) {
    stop(paste0("历史轨迹缺少指标字段：", field), call. = FALSE)
  }
  values <- suppressWarnings(as.numeric(history_spec$visits[[field]]))
  times <- suppressWarnings(as.numeric(history_spec$visits$time_years))
  valid <- is.finite(times) & is.finite(values)
  encoded_valid <- valid & history_spec$encoded
  background_valid <- valid & !history_spec$encoded
  observed <- data.frame(time = times[encoded_valid], value = values[encoded_valid])
  observed <- observed[order(observed$time), , drop = FALSE]
  background <- data.frame(time = times[background_valid], value = values[background_valid])
  background <- background[order(background$time), , drop = FALSE]

  individualized <- ddhigan_history_eb_curve(times, values, history_spec$encoded, model)

  list(
    field = field, label = label, color = color, pch = pch,
    background = background, observed = observed, smooth = individualized$curve,
    method = individualized$method, conditioning_n = individualized$conditioning_n
  )
}

ddhigan_history_series <- function(history_spec, lmm_artifact) {
  list(
    proteinuria = ddhigan_marker_series(
      history_spec, "proteinuria_g_24h", "PRO24H (g/24h)", "#0000FF", 16,
      lmm_artifact$markers$PRO24H
    ),
    creatinine = ddhigan_marker_series(
      history_spec, "creatinine_mg_dl", "CREA (mg/dL)", "#0000FF", 16,
      lmm_artifact$markers$CREA
    ),
    cystatin_c = ddhigan_marker_series(
      history_spec, "cystatin_c_mg_l", "Cystatin C (mg/L)", "#03BF3D", 17,
      lmm_artifact$markers$CystatinC
    )
  )
}

ddhigan_history_ylim <- function(series) {
  values <- unlist(lapply(series, function(item) c(
    item$background$value, item$observed$value, item$smooth$value
  )), use.names = FALSE)
  values <- values[is.finite(values)]
  if (!length(values)) return(c(0, 1))
  limits <- c(max(0, min(values) * 0.7), max(values) * 1.2)
  if (!all(is.finite(limits)) || diff(limits) <= 0) {
    centre <- if (length(values)) values[1] else 0
    padding <- max(abs(centre) * 0.2, 0.5)
    limits <- c(max(0, centre - padding), centre + padding)
  }
  limits
}

ddhigan_draw_history_panel <- function(
    series, history_spec, ylab, show_xlab = TRUE, xlab = "活检后年数") {
  graphics::plot(
    NA_real_, NA_real_, type = "n", axes = FALSE, xaxs = "i", yaxs = "i", bty = "n",
    xlim = history_spec$xlim, ylim = ddhigan_history_ylim(series),
    xlab = "", ylab = ""
  )
  x_ticks <- graphics::axTicks(1)
  x_ticks <- x_ticks[x_ticks >= 0]
  graphics::axis(
    1, at = x_ticks, labels = if (show_xlab) x_ticks else rep("", length(x_ticks)),
    cex.axis = 1, tcl = if (show_xlab) -0.3 else 0, col = "black", col.axis = "black"
  )
  graphics::axis(2, las = 1, cex.axis = 1, tcl = -0.3, col = "black", col.axis = "black")
  graphics::box(col = "black", lwd = 0.8)
  graphics::mtext(ylab, side = 2, line = 2.15, cex = 0.78, las = 0, col = "black")
  if (show_xlab) {
    graphics::mtext(xlab, side = 1, line = 1.65, cex = 0.9, col = "black")
  }
  graphics::abline(v = history_spec$query_time_years, col = "black", lty = 3, lwd = 1)

  for (item in series) {
    if (nrow(item$background)) {
      graphics::points(
        item$background$time, item$background$value,
        pch = item$pch, col = "#BDBDBD", bg = "#BDBDBD", cex = 0.8
      )
    }
  }
  for (item in series) {
    if (nrow(item$observed)) {
      graphics::points(
        item$observed$time, item$observed$value,
        pch = item$pch, col = item$color, bg = item$color, cex = 1
      )
    }
    if (identical(item$method, "lmm")) {
      graphics::lines(item$smooth$time, item$smooth$value, col = item$color, lwd = 2)
    } else if (nrow(item$observed) >= 2L) {
      graphics::lines(item$observed$time, item$observed$value, col = item$color, lwd = 2)
    }
  }
}

ddhigan_draw_history_plots <- function(history_spec, lmm_artifact, language = "zh") {
  language <- if (identical(language, "en")) "en" else "zh"
  xlab <- if (language == "en") "Years after biopsy" else "活检后年数"
  series <- ddhigan_history_series(history_spec, lmm_artifact)
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(
    mfrow = c(2, 1), oma = c(0.2, 0.2, 0.2, 0.2),
    mgp = c(2, 0.4, 0), tcl = -0.3, las = 0, bty = "o", bg = "white"
  )
  graphics::par(mar = c(0.35, 3.35, 0.25, 0.35))
  ddhigan_draw_history_panel(
    list(series$proteinuria), history_spec,
    "PRO24H", show_xlab = FALSE
  )
  graphics::par(mar = c(2.7, 3.35, 0.25, 0.35))
  ddhigan_draw_history_panel(
    list(series$creatinine), history_spec,
    "CREA", show_xlab = TRUE, xlab = xlab
  )
  invisible(series)
}

ddhigan_prepare_risk_series <- function(curve) {
  years <- suppressWarnings(as.numeric(curve$years))
  risk <- suppressWarnings(as.numeric(curve$risk))
  valid <- length(years) == 10L && identical(years, as.numeric(1:10)) &&
    length(risk) == length(years) && all(is.finite(risk)) &&
    all(risk >= 0 & risk <= 1) && all(diff(risk) >= -1e-12)
  if (!valid) stop("Risk curve is unavailable or invalid.", call. = FALSE)
  list(api_years = years, api_risk = risk, years = c(0, years), risk = c(0, risk))
}

ddhigan_draw_risk_plot <- function(risk_series, language = "zh") {
  language <- if (identical(language, "en")) "en" else "zh"
  xlab <- if (language == "en") "Years after prediction time" else "预测时点后的年数"
  ylab <- if (language == "en") "Cumulative kidney failure (ESKD) risk" else "累计 kidney failure (ESKD) 风险"
  title <- if (language == "en") "Future 10-year cumulative risk" else "未来 10 年累计风险"
  old_par <- graphics::par(no.readonly = TRUE)
  on.exit(graphics::par(old_par), add = TRUE)
  graphics::par(mar = c(4.2, 4.8, 2.2, 1.0), mgp = c(2.7, 0.8, 0), las = 1, bty = "l")
  graphics::plot(
    risk_series$years, risk_series$risk, type = "n", axes = FALSE,
    xlab = xlab, ylab = ylab, main = title,
    xlim = c(0, 10), ylim = c(0, 1),
    xaxs = "i", yaxs = "i"
  )
  graphics::abline(h = seq(0, 1, by = 0.1), col = "#ECEFF1", lwd = 0.8)
  graphics::abline(v = 0:10, col = "#F2F3F4", lwd = 0.8)
  graphics::axis(1, at = 0:10)
  graphics::axis(2, at = seq(0, 1, by = 0.1), labels = paste0(seq(0, 100, by = 10), "%"))
  graphics::lines(risk_series$years, risk_series$risk, lwd = 2.5, col = "#D62728")
  graphics::box(col = "#B8BEC4")
}
