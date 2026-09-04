app_source <- paste(readLines(file.path("..", "..", "app.R"), warn = FALSE, encoding = "UTF-8"), collapse = "\n")
client_source <- paste(readLines(file.path("..", "..", "R", "ddhigan_client.R"), warn = FALSE, encoding = "UTF-8"), collapse = "\n")
plot_source <- paste(readLines(file.path("..", "..", "R", "ddhigan_plot.R"), warn = FALSE, encoding = "UTF-8"), collapse = "\n")

testthat::test_that("web input is identifier free and bounded", {
  testthat::expect_false(grepl("patient_id|patientId|mrn|medical_record", app_source, ignore.case = TRUE))
  testthat::expect_match(app_source, "请勿输入姓名、病案号、日期或其他身份信息", fixed = TRUE)
  testthat::expect_match(app_source, "当前为公开互联网部署", fixed = TRUE)
  testthat::expect_match(app_source, "public-internet deployment", fixed = TRUE)
  testthat::expect_false(grepl("当前为局域网部署|This is a LAN deployment", app_source))
  testthat::expect_match(app_source, "nrow(visits) > 256L", fixed = TRUE)
  testthat::expect_match(app_source, "visits$time_years[[1L]] != 0", fixed = TRUE)
  testthat::expect_match(app_source, "is.unsorted(visits$time_years, strictly = TRUE)", fixed = TRUE)
})

testthat::test_that("biopsy labs and follow-up visits share one bilingual table", {
  testthat::expect_match(app_source, 'class = "input-table visit-table"', fixed = TRUE)
  testthat::expect_match(app_source, '"language"', fixed = TRUE)
  testthat::expect_match(app_source, '"中文" = "zh"', fixed = TRUE)
  testthat::expect_match(app_source, '"English" = "en"', fixed = TRUE)
  testthat::expect_match(app_source, 'parseQueryString(session$clientData$url_search)', fixed = TRUE)
  testthat::expect_match(app_source, 'Add visit', fixed = TRUE)
  testthat::expect_match(app_source, '新增访视', fixed = TRUE)
  testthat::expect_match(app_source, 'selected = "en"', fixed = TRUE)
  testthat::expect_match(app_source, 'text$at_biopsy', fixed = TRUE)
  testthat::expect_match(app_source, 'get_col("visit_time_", 0)', fixed = TRUE)
  testthat::expect_match(app_source, 'creatinine_mg_dl = visits$creatinine_mg_dl[[1L]]', fixed = TRUE)
  testthat::expect_false(grepl('numericInput("base_crea"', app_source, fixed = TRUE))
  testthat::expect_false(grepl('numericInput("base_cysc"', app_source, fixed = TRUE))
  testthat::expect_false(grepl('numericInput("base_alb"', app_source, fixed = TRUE))
  testthat::expect_false(grepl('numericInput("base_pro"', app_source, fixed = TRUE))
  testthat::expect_false(grepl("textAreaInput", app_source, fixed = TRUE))
  testthat::expect_false(grepl("parse_visit_csv", app_source, fixed = TRUE))
})

testthat::test_that("both languages are passed into history and risk plots", {
  testthat::expect_match(app_source, "ddhigan_draw_history_plots(result()$history, result()$lmm, lang())", fixed = TRUE)
  testthat::expect_match(app_source, "ddhigan_draw_risk_plot(result()$risk, lang())", fixed = TRUE)
  testthat::expect_match(plot_source, "Future 10-year cumulative risk", fixed = TRUE)
  draw_start <- regexpr("ddhigan_draw_history_plots <-", plot_source, fixed = TRUE)[1]
  draw_end <- regexpr("ddhigan_prepare_risk_series <-", plot_source, fixed = TRUE)[1]
  history_draw <- substr(plot_source, draw_start, draw_end - 1L)
  testthat::expect_match(history_draw, 'list(series$creatinine)', fixed = TRUE)
  testthat::expect_false(grepl("series$cystatin_c", history_draw, fixed = TRUE))
})

testthat::test_that("API key remains server side and HTTPS CA verification is enabled", {
  testthat::expect_false(grepl("DDHIGAN_API_KEY=", app_source, fixed = TRUE))
  testthat::expect_match(client_source, "X-API-Key", fixed = TRUE)
  testthat::expect_match(client_source, "cainfo", fixed = TRUE)
  testthat::expect_match(app_source, 'Sys.getenv("DDHIGAN_WEB_ENV_FILE", unset = "")', fixed = TRUE)
  testthat::expect_match(app_source, "DDHIGAN_HISTORY_LMM_PATH", fixed = TRUE)
  testthat::expect_false(grepl("DDHIGAN_API_KEY=", app_source, fixed = TRUE))
})

testthat::test_that("web disclosure preserves model separation", {
  testthat::expect_match(app_source, "不参与 DDHIgAN 风险推理", fixed = TRUE)
  testthat::expect_match(app_source, "pdDiag", fixed = TRUE)
  testthat::expect_match(app_source, "不用于因果病程解释", fixed = TRUE)
  testthat::expect_match(app_source, "该患者风险预测的 95% 点态 bootstrap 区间", fixed = TRUE)
  testthat::expect_match(app_source, "unmeasured factors, model misspecification", fixed = TRUE)
  testthat::expect_false(grepl("不是患者真实风险的置信区间", app_source, fixed = TRUE))
})
