# Audit clone: same history routing, only retain t0 and expose pre-conditioning S(t0).
# Repeat t0 to retain the package matrix shape when aggregating multiple subjects.
# Does not modify the package namespace or fitted forest.
dynforest_denominator_function <- function() {
 stopifnot(as.character(utils::packageVersion('DynForest'))=='1.2.0')
 original<-get('predict.dynforest',asNamespace('DynForest'))
 src<-paste(deparse(body(original)),collapse='\n')
 needle<-'predTimes <- c(t0, allTimes[which(allTimes >= t0)])'
 stopifnot(length(gregexpr(needle,src,fixed=TRUE)[[1]])==1L,grepl(needle,src,fixed=TRUE))
 src<-sub(needle,'predTimes <- c(t0, t0)',src,fixed=TRUE)
 needle<-'class(output) <- c("dynforestpred")'
 stopifnot(grepl(needle,src,fixed=TRUE))
 src<-sub(needle,paste0('output$query_denominator <- 1 - Reduce("+", lapply(pred_cif_mean, function(z) z[,1]));\n',needle),src,fixed=TRUE)
 audited<-original;body(audited)<-parse(text=src)[[1]]
 audited
}

summarize_dynforest_denominator <- function(model,timeData,fixedData,q) {
 stopifnot(model$type=='surv',length(model$causes)==1L,is.finite(q),q>=0,q<=5,max(model$times)>=q+10)
 fun<-dynforest_denominator_function()
 p<-fun(model,timeData=timeData,fixedData=fixedData,idVar='patient_id',timeVar='visit_time',t0=q)
 den<-p$query_denominator
 stopifnot(length(den)==nrow(fixedData),all(is.finite(den)),all(den>0 & den<=1))
 list(n=length(den),minimum=min(den),below_1e12=sum(den<1e-12),below_1e6=sum(den<1e-6))
}
