# Stable residual Cox conditioning; package's cumulative hazard is a step function.
conditional_cox_survival <- function(baseline_times, baseline_hazard, lp, delay, horizons, support) {
  stopifnot(length(baseline_times)==length(baseline_hazard),length(lp)==length(delay),
            length(lp)>0L,length(horizons)>0L,length(baseline_times)>0L,
            all(is.finite(c(baseline_times,baseline_hazard,lp,delay,horizons,support))),
            all(diff(baseline_times)>=0),all(diff(baseline_hazard)>=0),
            all(baseline_hazard>=0),all(delay>=0),all(horizons>0),
            max(outer(delay,horizons,'+'))<=support)
  at<-function(t)c(0,baseline_hazard)[findInterval(t,baseline_times)+1L]
  delta<-vapply(horizons,function(h)at(delay+h)-at(delay),numeric(length(delay)))
  delta<-matrix(delta,nrow=length(delay),ncol=length(horizons))
  stopifnot(all(delta>=0))
  # log(delta H) + lp avoids 0*Inf and preserves exact zero hazard increments.
  log_increment<-log(delta)+lp
  survival<-exp(-exp(log_increment))
  stopifnot(all(is.finite(survival)),all(survival>=0 & survival<=1))
  survival
}

predict_pccox_query <- function(model, frame, horizons=1:10) {
  stopifnot(inherits(model,'PC_cox'),all(c('query_time','last_observation_time','measurement_time','prediction_delay') %in% names(frame)),
            all(frame$measurement_time==frame$last_observation_time),
            all(frame$query_time>=0 & frame$query_time<=5),
            all(abs(frame$prediction_delay-(frame$query_time-frame$last_observation_time))<1e-14))
  frame$stime<-frame$duration+frame$query_time;frame$status<-frame$event
  fit<-model$model.fit
  lp<-as.numeric(predict(fit,newdata=frame,type='lp',reference='sample'))
  bh<-survival::basehaz(fit,centered=TRUE)
  survival<-conditional_cox_survival(bh$time,bh$hazard,lp,frame$prediction_delay,horizons,max(fit$y[,1]))
  out<-frame[c('patient_id','query_time','last_observation_time','prediction_delay')]
  out$true_time<-frame$stime;out$residual_time<-frame$duration;out$true_event<-frame$event;out$Tstart<-frame$query_time
  for(i in seq_along(horizons))out[[sprintf('resid_surv_%.1fy',horizons[i])]]<-survival[,i]
  for(h in c(3,5,10))out[[paste0('cond_surv_',h,'y')]]<-out[[sprintf('resid_surv_%.1fy',h)]]
  out$risk_score<-1-out$cond_surv_10y
  out
}
