options(shiny.maxRequestSize = 1 * 1024^2)

env_file <- Sys.getenv("DDHIGAN_WEB_ENV_FILE", "/etc/ddhigan-web/environment")
if (file.exists(env_file)) readRenviron(env_file)
if (!nzchar(Sys.getenv("DDHIGAN_HISTORY_LMM_PATH", unset = ""))) {
  Sys.setenv(DDHIGAN_HISTORY_LMM_PATH = file.path(
    "data", "ddhigan_history_lmm_physv11_20260904.rds"
  ))
}

suppressPackageStartupMessages({
  library(shiny)
  library(httr2)
  library(jsonlite)
  library(splines)
})

source("R/ddhigan_client.R", encoding = "UTF-8")
source("R/ddhigan_history_lmm.R", encoding = "UTF-8")
source("R/ddhigan_plot.R", encoding = "UTF-8")

required_visit_columns <- c(
  "time_years", "creatinine_mg_dl", "cystatin_c_mg_l",
  "albumin_g_l", "proteinuria_g_24h"
)

web_text <- list(
  zh = list(
    title = "DDHIgAN 动态肾衰竭风险研究工具",
    notice = "研究性临床试点，不替代临床判断。当前为公开互联网部署；请勿输入姓名、病案号、日期或其他身份信息。",
    baseline = "静态信息",
    age = "活检年龄", sex = "性别", female = "女", male = "男",
    visits = "活检时及活检后检测", visit_help = "首行固定为活检时（0 年），其检测值同时作为基线值；后续行填写活检后访视。缺失指标留空。API 仅编码最近 30 次，较早访视仍以灰点显示。",
    visit_no = "时点", at_biopsy = "活检时", post_biopsy = "活检后访视", time = "活检后时间", add = "新增访视", remove = "删除末行",
    confirm = "确认预测时点尚未发生 kidney failure (ESKD)", predict = "生成预测",
    history = "历史轨迹", risk = "未来 10 年风险", disclosure = "数据与模型说明",
    disclosure_1 = "自然样条线性混合模型仅用于描述性历史可视化，不参与 DDHIgAN 风险推理。随机样条系数采用原 IgAN-JM 的 pdDiag 独立协方差结构；观测较少时曲线主要收缩至总体趋势。",
    disclosure_2 = "历史曲线不向未来或训练范围外外推，也不校正信息性失访或结局前随访终止，不用于因果病程解释。风险曲线由独立的 DDHIgAN API 返回。",
    risk_note = "曲线为所选模型五折预测的平均值，未作额外再校准。此部署平均预测不同于内部验证中的单个留出折预测；不显示个体置信区间，也不把折间差异解释为个体不确定性。",
    need_confirm = "必须确认预测时点无 kidney failure (ESKD)。", calculating = "正在验证并计算……",
    visit_count = "访视数必须为 1–256。", time_error = "访视时间必须从 0 开始、非负并严格递增。",
    marker_error = "每次访视至少需要一个有限、非负指标值。", baseline_marker_error = "活检时首行的 CREA、Cystatin C、ALB 和 PRO24H 均为必填的非负数值。", static_error = "活检年龄必须为 0–120 岁的有限数值。",
    model_version = "模型版本：", api_version = "API 版本：", risk3 = "未来 3 年：", risk5 = "未来 5 年：", risk10 = "未来 10 年：",
    history_count = "收到 %d 次访视；编码最近 %d 次；灰点 %d 次。"
  ),
  en = list(
    title = "DDHIgAN Dynamic Kidney Failure Risk Research Tool",
    notice = "Research-use clinical pilot; not a substitute for clinical judgment. This is a public-internet deployment. Do not enter names, record numbers, dates, or other identifiers.",
    baseline = "Static information",
    age = "Age at biopsy", sex = "Sex", female = "Female", male = "Male",
    visits = "At-biopsy and post-biopsy measurements", visit_help = "The first row is fixed at biopsy (0 years), and its measurements also supply the baseline values. Add subsequent post-biopsy visits below. Leave unavailable markers blank. The API encodes the latest 30 visits; earlier visits remain visible as grey points.",
    visit_no = "Time point", at_biopsy = "At biopsy", post_biopsy = "Post-biopsy visit", time = "Time after biopsy", add = "Add visit", remove = "Remove last row",
    confirm = "Confirm that kidney failure (ESKD) has not occurred by the prediction time", predict = "Generate prediction",
    history = "History", risk = "Future 10-year risk", disclosure = "Data and model notes",
    disclosure_1 = "Natural-spline linear mixed models are used only for descriptive history visualization and do not enter DDHIgAN risk inference. Random spline coefficients use the original IgAN-JM pdDiag independent covariance structure; sparse histories shrink mainly toward the population trend.",
    disclosure_2 = "History curves are not extrapolated into the future or beyond the training range and do not correct informative dropout or follow-up termination before the endpoint. They are not causal disease-course estimates. The risk curve is returned independently by the DDHIgAN API.",
    risk_note = "The curve averages predictions from the five saved folds of the selected model, without additional recalibration. This deployment average differs from the held-out-fold predictor used in internal validation. No individual confidence interval is shown; fold variation is not an individual uncertainty interval.",
    need_confirm = "Confirm that kidney failure (ESKD) has not occurred by the prediction time.", calculating = "Validating inputs and calculating…",
    visit_count = "The visit table must contain 1–256 rows.", time_error = "Visit time must start at 0, be nonnegative, and increase strictly.",
    marker_error = "Every visit needs at least one finite, nonnegative marker value.", baseline_marker_error = "CREA, Cystatin C, ALB, and PRO24H are all required as nonnegative values in the at-biopsy row.", static_error = "Age at biopsy must be a finite value from 0 to 120 years.",
    model_version = "Model version: ", api_version = "API release: ", risk3 = "Next 3 years: ", risk5 = "Next 5 years: ", risk10 = "Next 10 years: ",
    history_count = "Received %d visits; encoded the latest %d; displayed %d earlier visits in grey."
  )
)

parse_visit_table <- function(visits, language = "zh") {
  text <- web_text[[if (identical(language, "en")) "en" else "zh"]]
  if (!is.data.frame(visits) || !identical(names(visits), required_visit_columns) ||
      !nrow(visits) || nrow(visits) > 256L) stop(text$visit_count, call. = FALSE)
  for (name in names(visits)) visits[[name]] <- suppressWarnings(as.numeric(visits[[name]]))
  if (any(!is.finite(visits$time_years)) || any(visits$time_years < 0) ||
      is.unsorted(visits$time_years, strictly = TRUE) || visits$time_years[[1L]] != 0) {
    stop(text$time_error, call. = FALSE)
  }
  marker_matrix <- as.matrix(visits[setdiff(names(visits), "time_years")])
  if (any(is.infinite(marker_matrix), na.rm = TRUE) ||
      any(marker_matrix < 0, na.rm = TRUE) ||
      any(rowSums(is.finite(marker_matrix)) == 0L)) {
    stop(text$marker_error, call. = FALSE)
  }
  visits
}

ui <- fluidPage(
  tags$head(
    tags$meta(charset = "utf-8"),
    tags$meta(name = "viewport", content = "width=device-width, initial-scale=1"),
    tags$style(HTML(
      "body{background:#f7f9fb;color:#202124;font-family:Arial,'Microsoft YaHei',sans-serif}
       .container-fluid{max-width:1180px;margin:auto}.card{background:#fff;border:1px solid #e2e6ea;
       border-radius:10px;padding:18px;margin:14px 0;box-shadow:0 1px 3px rgba(0,0,0,.05)}
       h2{color:#176b75}.notice{background:#fff8e1;border-left:4px solid #f9a825;padding:10px 14px}
       .language-bar{display:flex;justify-content:flex-end;margin-top:10px}.language-bar .form-group{margin:0}
       .input-table-wrap{overflow-x:auto}.input-table{width:100%;border-collapse:collapse;table-layout:fixed}
       .input-table th,.input-table td{border:1px solid #dfe3e7;padding:7px 9px;vertical-align:middle}
       .input-table th{background:#eef4f6;color:#24464b;text-align:left}.input-table .form-group{margin:0}
       .input-table input,.input-table select{width:100%}.visit-table{min-width:820px}
       .table-actions{display:flex;gap:8px;margin-top:10px}.shiny-output-error{white-space:normal}"
    ))
  ),
  div(class = "language-bar", radioButtons(
    "language", NULL, choices = c("中文" = "zh", "English" = "en"),
    selected = "en", inline = TRUE
  )),
  uiOutput("page")
)

server <- function(input, output, session) {
  result <- reactiveVal(NULL)
  status <- reactiveVal("")
  visit_rows <- reactiveVal(2L)
  lang <- reactive(if (identical(input$language, "zh")) "zh" else "en")
  txt <- reactive(web_text[[lang()]])

  observeEvent(session$clientData$url_search, {
    requested <- parseQueryString(session$clientData$url_search)[["lang"]]
    if (length(requested) == 1L && requested %in% c("zh", "en")) {
      updateRadioButtons(session, "language", selected = requested)
    }
  }, once = TRUE, ignoreInit = FALSE)

  current_value <- function(id, default) {
    value <- isolate(input[[id]])
    if (is.null(value) || !length(value) || is.na(value)) default else value
  }
  visit_input <- function(prefix, row, default) {
    numericInput(paste0(prefix, row), NULL, current_value(paste0(prefix, row), default), min = 0)
  }

  output$page <- renderUI({
    text <- txt()
    tagList(
      titlePanel(text$title), div(class = "notice", text$notice),
      div(class = "card", selectInput("model_id", if (lang() == "en") "Prediction model" else "预测模型", choices = c("DDHIgAN", "DDHIgAN-CysC-free"), selected = current_value("model_id", "DDHIgAN"))),
      div(class = "card", h4(text$baseline), fluidRow(
        column(6, numericInput("age", paste0(text$age, if (lang() == "en") " (years)" else "（岁）"), current_value("age", 40), min = 0, max = 120)),
        column(6, selectInput("sex", text$sex, c(setNames("female", text$female), setNames("male", text$male)), selected = current_value("sex", "female")))
      )),
      div(class = "card", h4(text$visits), p(text$visit_help), uiOutput("visit_table"),
        div(class = "table-actions", actionButton("add_visit", text$add), actionButton("remove_visit", text$remove))
      ),
      div(class = "card", checkboxInput("eskd_free", text$confirm, current_value("eskd_free", TRUE)),
          actionButton("predict", text$predict, class = "btn-primary")),
      uiOutput("status"),
      conditionalPanel("output.has_result",
        fluidRow(
          column(4, div(class = "card", h4(text$history), plotOutput("history", height = "520px"))),
          column(8, div(class = "card", h4(text$risk), plotOutput("risk", height = "480px"), uiOutput("summary")))
        ),
        div(class = "card", h4(text$disclosure), p(text$risk_note), p(text$disclosure_1), p(text$disclosure_2))
      )
    )
  })

  output$visit_table <- renderUI({
    text <- txt(); n <- visit_rows()
    defaults <- list(time = c(0, 1), crea = c(1.0, 1.1), cysc = c(1.0, 1.1), alb = c(40, 38), pro = c(1.0, 1.3))
    default_at <- function(name, row) if (row <= 2L) defaults[[name]][row] else if (name == "time") row - 1 else defaults[[name]][2]
    rows <- lapply(seq_len(n), function(row) tags$tr(
      tags$td(if (row == 1L) text$at_biopsy else paste(text$post_biopsy, row - 1L)),
      tags$td(if (row == 1L) tags$span(class = "fixed-zero", "0") else visit_input("visit_time_", row, default_at("time", row))),
      tags$td(visit_input("visit_crea_", row, default_at("crea", row))),
      if (!identical(input$model_id, "DDHIgAN-CysC-free")) tags$td(visit_input("visit_cysc_", row, default_at("cysc", row))),
      tags$td(visit_input("visit_alb_", row, default_at("alb", row))),
      tags$td(visit_input("visit_pro_", row, default_at("pro", row)))
    ))
    div(class = "input-table-wrap", tags$table(class = "input-table visit-table",
      tags$thead(tags$tr(
        tags$th(text$visit_no), tags$th(paste0(text$time, " (years)")),
        tags$th("CREA (mg/dL)"), if (!identical(input$model_id, "DDHIgAN-CysC-free")) tags$th("Cystatin C (mg/L)"),
        tags$th("ALB (g/L)"), tags$th("PRO24H (g/24h)")
      )), tags$tbody(rows)
    ))
  })

  observeEvent(input$add_visit, visit_rows(min(256L, visit_rows() + 1L)))
  observeEvent(input$remove_visit, visit_rows(max(1L, visit_rows() - 1L)))

  collect_visits <- function() {
    n <- visit_rows()
    get_col <- function(prefix, first_value = NULL) vapply(seq_len(n), function(row) {
      if (row == 1L && !is.null(first_value)) return(first_value)
      value <- input[[paste0(prefix, row)]]
      if (is.null(value) || !length(value)) NA_real_ else suppressWarnings(as.numeric(value))
    }, numeric(1))
    data.frame(
      time_years = get_col("visit_time_", 0), creatinine_mg_dl = get_col("visit_crea_"),
      cystatin_c_mg_l = if (identical(input$model_id, "DDHIgAN-CysC-free")) rep(NA_real_, n) else get_col("visit_cysc_"), albumin_g_l = get_col("visit_alb_"),
      proteinuria_g_24h = get_col("visit_pro_"), check.names = FALSE
    )
  }

  observeEvent(input$model_id, { result(NULL); status("") }, ignoreInit = TRUE)

  observeEvent(input$predict, {
    text <- txt()
    result(NULL)
    if (!isTRUE(input$eskd_free)) {
      status(text$need_confirm)
      return()
    }
    status(text$calculating)
    candidate <- tryCatch({
      visits <- parse_visit_table(collect_visits(), lang())
      static <- list(
        age_at_biopsy_years = as.numeric(input$age), sex = input$sex,
        creatinine_mg_dl = visits$creatinine_mg_dl[[1L]],
        cystatin_c_mg_l = visits$cystatin_c_mg_l[[1L]],
        albumin_g_l = visits$albumin_g_l[[1L]],
        proteinuria_g_24h = visits$proteinuria_g_24h[[1L]]
      )
      if (!is.finite(static$age_at_biopsy_years) || static$age_at_biopsy_years < 0 || static$age_at_biopsy_years > 120) stop(text$static_error, call. = FALSE)
      fields <- c("creatinine_mg_dl", "albumin_g_l", "proteinuria_g_24h")
      if (!identical(input$model_id, "DDHIgAN-CysC-free")) fields <- c(fields, "cystatin_c_mg_l")
      baseline_markers <- unlist(static[fields], use.names = FALSE)
      if (any(!is.finite(baseline_markers)) || any(baseline_markers < 0)) stop(text$baseline_marker_error, call. = FALSE)
      prediction <- ddhigan_predict(static, visits, tail(visits$time_years, 1L), model_id = input$model_id)
      history_spec <- ddhigan_history_plot_spec(
        visits, prediction$history_processing, tail(visits$time_years, 1L)
      )
      list(
        prediction = prediction,
        history = history_spec,
        lmm = ddhigan_load_history_lmm(),
        risk = ddhigan_prepare_risk_series(prediction$future_risk_curve)
      )
    }, error = function(error) error)
    if (inherits(candidate, "error")) {
      status(conditionMessage(candidate))
      return()
    }
    result(candidate)
    status("")
  })

  output$has_result <- reactive(!is.null(result()))
  outputOptions(output, "has_result", suspendWhenHidden = FALSE)
  output$status <- renderUI(if (nzchar(status())) div(class = "alert alert-info", status()))
  output$history <- renderPlot({
    req(result())
    ddhigan_draw_history_plots(result()$history, result()$lmm, lang())
  }, res = 120)
  output$risk <- renderPlot({
    req(result())
    ddhigan_draw_risk_plot(result()$risk, lang())
  }, res = 120)
  output$summary <- renderUI({
    req(result())
    text <- txt()
    prediction <- result()$prediction
    tagList(
      p(strong(if (lang() == "en") "Selected model: " else "所选模型："), prediction$model_id),
      p(strong(text$model_version), prediction$model_version),
      if (isTRUE(prediction$warnings$out_of_distribution)) p(prediction$warnings$late_followup_extrapolation),
      p(strong(text$api_version), prediction$api_release),
      p(strong(text$risk3), scales::percent(prediction$risk_3y, accuracy = 0.1)),
      p(strong(text$risk5), scales::percent(prediction$risk_5y, accuracy = 0.1)),
      p(strong(text$risk10), scales::percent(prediction$risk_10y, accuracy = 0.1)),
      p(sprintf(text$history_count,
                prediction$history_processing$received_visits,
                prediction$history_processing$encoded_visits,
                prediction$history_processing$omitted_visits))
    )
  })
}

shinyApp(ui, server)
