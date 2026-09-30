# Saved-fit inference only; explicit independent output namespace.
if(nzchar(Sys.getenv("PCCOX_R_LIBRARY"))) .libPaths(c(Sys.getenv("PCCOX_R_LIBRARY"),.libPaths()))
suppressPackageStartupMessages({library(partlyconditional);library(survival);library(jsonlite)})
source('R/query_time_survival.R')
a<-commandArgs(TRUE);stopifnot(length(a)==4L)
model_path<-a[1];input<-a[2];output<-a[3];fold<-as.integer(a[4]);stopifnot(fold %in% 0:4,!dir.exists(output))
model<-readRDS(model_path);dir.create(output,recursive=TRUE)
for(q in 0:5){
 frame<-read.csv(file.path(input,paste0('test_landmark',q,'.csv')),colClasses=c(patient_id='character'))
 stopifnot(all(frame$query_time==q),!anyDuplicated(frame$patient_id))
 result<-predict_pccox_query(model,frame)
 stopifnot(nrow(result)==nrow(frame))
 write.csv(result,file.path(output,sprintf('pccox_query15_ESKD_fold%d_test_landmark%d.csv',fold,q)),row.names=FALSE)
}
write_json(list(status='INFERENCE_COMPLETE',fit_count=0,fold=fold),file.path(output,'inference.json'),auto_unbox=TRUE)
