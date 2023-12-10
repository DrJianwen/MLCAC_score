#Part_1
library(glmnet)
library(xlsx)
#Import the train data with 54 variables, and this part of original data are available from the corresponding author on reasonable request.
data<-read.xlsx("D:/Train_data_Z.xlsx",1)
#LASSO lambda selection and the corresponding mean AUC using 10-fold cross-validation:
cvfit <- cv.glmnet(x=model.matrix(~.,data[,-c(1)]),
                   y=data$CT.CACscore_above400,
                   family = "binomial",
                   grouped=FALSE,
                   nfolds = 10,
                   set.seed(42),
                   nlambda = 100,
                   alpha=1,
                   type.measure = c("auc"))
plot(cvfit)
abline(v=log(0.0318), col="black", lty=3) #plot Of 9 features selected 
#LASSO coefficient profiles of the potential predictors:
lasso = glmnet(x=model.matrix(~.,data[,-c(1)]),
               y=data$CT.CACscore_above400,
               family = "binomial",
               alpha=1)
plot(lasso, xvar = "lambda")
abline(v=log(cvfit$lambda.1se), col="black", lty=3 )
abline(v=log(cvfit$lambda.min), col="black", lty=3 )
abline(v=log(0.0318), col="black", lty=3 ) #plot Of 9 features selected
#two sets of features selected
coef(cvfit, s = "lambda.1se") #lambda.1se selected
coef(cvfit, s = 0.0318) #9 features selected





#Part_2
#DCA analysis
library(xlsx)
data<-read.xlsx("D:/DCA_final_results.xlsx",1)
data <- as.data.frame(data)
dca <- function(data, outcome, predictors, xstart=0.01, xstop=0.99, xby=0.01, 
                ymin=-0.05, probability=NULL, harm=NULL,graph=TRUE, intervention=FALSE, 
                interventionper=100, smooth=FALSE,loess.span=0.10) {
  # LOADING REQUIRED LIBRARIES
  require(stats)
  
  # data MUST BE A DATA FRAME
  if (class(data)!="data.frame") {
    stop("Input data must be class data.frame")
  }
  
  #ONLY KEEPING COMPLETE CASES
  data=data[complete.cases(data[append(outcome,predictors)]),append(outcome,predictors)]
  
  # outcome MUST BE CODED AS 0 AND 1
  if (max(data[[outcome]])>1 | min(data[[outcome]])<0) {
    stop("outcome cannot be less than 0 or greater than 1")
  }
  # xstart IS BETWEEN 0 AND 1
  if (xstart<0 | xstart>1) {
    stop("xstart must lie between 0 and 1")
  }
  
  # xstop IS BETWEEN 0 AND 1
  if (xstop<0 | xstop>1) {
    stop("xstop must lie between 0 and 1")
  }
  
  # xby IS BETWEEN 0 AND 1
  if (xby<=0 | xby>=1) {
    stop("xby must lie between 0 and 1")
  }
  
  # xstart IS BEFORE xstop
  if (xstart>=xstop) {
    stop("xstop must be larger than xstart")
  }
  
  #STORING THE NUMBER OF PREDICTORS SPECIFIED
  pred.n=length(predictors)
  
  #IF probability SPECIFIED ENSURING THAT EACH PREDICTOR IS INDICATED AS A YES OR NO
  if (length(probability)>0 & pred.n!=length(probability)) {
    stop("Number of probabilities specified must be the same as the number of predictors being checked.")
  }
  
  #IF harm SPECIFIED ENSURING THAT EACH PREDICTOR HAS A SPECIFIED HARM
  if (length(harm)>0 & pred.n!=length(harm)) {
    stop("Number of harms specified must be the same as the number of predictors being checked.")
  }
  
  #INITIALIZING DEFAULT VALUES FOR PROBABILITES AND HARMS IF NOT SPECIFIED
  if (length(harm)==0) {
    harm=rep(0,pred.n)
  }
  if (length(probability)==0) {
    probability=rep(TRUE,pred.n)
  }
  
  
  #CHECKING THAT EACH probability ELEMENT IS EQUAL TO YES OR NO, 
  #AND CHECKING THAT PROBABILITIES ARE BETWEEN 0 and 1
  #IF NOT A PROB THEN CONVERTING WITH A LOGISTIC REGRESSION
  for(m in 1:pred.n) { 
    if (probability[m]!=TRUE & probability[m]!=FALSE) {
      stop("Each element of probability vector must be TRUE or FALSE")
    }
    if (probability[m]==TRUE & (max(data[predictors[m]])>1 | min(data[predictors[m]])<0)) {
      stop(paste(predictors[m],"must be between 0 and 1 OR sepcified as a non-probability in the probability option",sep=" "))  
    }
    if(probability[m]==FALSE) {
      model=NULL
      pred=NULL
      model=glm(data.matrix(data[outcome]) ~ data.matrix(data[predictors[m]]), family=binomial("logit"))
      pred=data.frame(model$fitted.values)
      pred=data.frame(pred)
      names(pred)=predictors[m]
      data=cbind(data[names(data)!=predictors[m]],pred)
      print(paste(predictors[m],"converted to a probability with logistic regression. Due to linearity assumption, miscalibration may occur.",sep=" "))
    }
  }
  
  # THE PREDICTOR NAMES CANNOT BE EQUAL TO all OR none.
  if (length(predictors[predictors=="all" | predictors=="none"])) {
    stop("Prediction names cannot be equal to all or none.")
  }  
  
  #########  CALCULATING NET BENEFIT   #########
  N=dim(data)[1]
  event.rate=colMeans(data[outcome])
  
  # CREATING DATAFRAME THAT IS ONE LINE PER THRESHOLD PER all AND none STRATEGY
  nb=data.frame(seq(from=xstart, to=xstop, by=xby))
  names(nb)="threshold"
  interv=nb
  
  nb["all"]=event.rate - (1-event.rate)*nb$threshold/(1-nb$threshold)
  nb["none"]=0
  
  # CYCLING THROUGH EACH PREDICTOR AND CALCULATING NET BENEFIT
  for(m in 1:pred.n){
    for(t in 1:length(nb$threshold)){
      # COUNTING TRUE POSITIVES AT EACH THRESHOLD
      tp=mean(data[data[[predictors[m]]]>=nb$threshold[t],outcome])*sum(data[[predictors[m]]]>=nb$threshold[t])
      # COUNTING FALSE POSITIVES AT EACH THRESHOLD
      fp=(1-mean(data[data[[predictors[m]]]>=nb$threshold[t],outcome]))*sum(data[[predictors[m]]]>=nb$threshold[t])
      #setting TP and FP to 0 if no observations meet threshold prob.
      if (sum(data[[predictors[m]]]>=nb$threshold[t])==0) {
        tp=0
        fp=0
      }
      
      # CALCULATING NET BENEFIT
      nb[t,predictors[m]]=tp/N - fp/N*(nb$threshold[t]/(1-nb$threshold[t])) - harm[m]
    }
    interv[predictors[m]]=(nb[predictors[m]] - nb["all"])*interventionper/(interv$threshold/(1-interv$threshold))
  }
  
  # CYCLING THROUGH EACH PREDICTOR AND SMOOTH NET BENEFIT AND INTERVENTIONS AVOIDED 
  for(m in 1:pred.n) {
    if (smooth==TRUE){
      lws=loess(data.matrix(nb[!is.na(nb[[predictors[m]]]),predictors[m]]) ~ data.matrix(nb[!is.na(nb[[predictors[m]]]),"threshold"]),span=loess.span)
      nb[!is.na(nb[[predictors[m]]]),paste(predictors[m],"_sm",sep="")]=lws$fitted
      
      lws=loess(data.matrix(interv[!is.na(nb[[predictors[m]]]),predictors[m]]) ~ data.matrix(interv[!is.na(nb[[predictors[m]]]),"threshold"]),span=loess.span)
      interv[!is.na(nb[[predictors[m]]]),paste(predictors[m],"_sm",sep="")]=lws$fitted
    }
  }
  
  # PLOTTING GRAPH IF REQUESTED
  if (graph==TRUE) {
    require(graphics)
    
    # PLOTTING INTERVENTIONS AVOIDED IF REQUESTED
    if(intervention==TRUE) {
      # initialize the legend label, color, and width using the standard specs of the none and all lines
      legendlabel <- NULL
      legendcolor <- NULL
      legendwidth <- NULL
      legendpattern <- NULL
      
      #getting maximum number of avoided interventions
      ymax=max(interv[predictors],na.rm = TRUE)
      
      #INITIALIZING EMPTY PLOT WITH LABELS
      plot(x=nb$threshold, y=nb$all, type="n" ,xlim=c(xstart, xstop), ylim=c(ymin, ymax), xlab="Threshold probability", ylab=paste("Net reduction in interventions per",interventionper,"patients"))
      
      #PLOTTING INTERVENTIONS AVOIDED FOR EACH PREDICTOR
      for(m in 1:pred.n) {
        if (smooth==TRUE){
          lines(interv$threshold,data.matrix(interv[paste(predictors[m],"_sm",sep="")]),col=m,lty=2)
        } else {
          lines(interv$threshold,data.matrix(interv[predictors[m]]),col=m,lty=2)
        }
        
        # adding each model to the legend
        legendlabel <- c(legendlabel, predictors[m])
        legendcolor <- c(legendcolor, m)
        legendwidth <- c(legendwidth, 1)
        legendpattern <- c(legendpattern, 2)
      }
    } else {
      # PLOTTING NET BENEFIT IF REQUESTED
      
      # initialize the legend label, color, and width using the standard specs of the none and all lines
      legendlabel <- c("None", "All")
      legendcolor <- c(17, 8)
      legendwidth <- c(2, 2)
      legendpattern <- c(1, 1)
      
      #getting maximum net benefit
      ymax=max(nb[names(nb)!="threshold"],na.rm = TRUE)
      
      # inializing new benfit plot with treat all option
      plot(x=nb$threshold, y=nb$all, type="l", col=8, lwd=2 ,xlim=c(xstart, xstop), ylim=c(ymin, ymax), xlab="Threshold probability", ylab="Net benefit")
      # adding treat none option
      lines(x=nb$threshold, y=nb$none,lwd=2)
      #PLOTTING net benefit FOR EACH PREDICTOR
      for(m in 1:pred.n) {
        if (smooth==TRUE){
          lines(nb$threshold,data.matrix(nb[paste(predictors[m],"_sm",sep="")]),col=m,lty=2,lwd = 1.5) 
        } else {
          lines(nb$threshold,data.matrix(nb[predictors[m]]),col=m,lty=2,lwd = 1.5)
        }
        # adding each model to the legend
        legendlabel <- c(legendlabel, predictors[m])
        legendcolor <- c(legendcolor, m)
        legendwidth <- c(legendwidth, 1.5)
        legendpattern <- c(legendpattern, 2)
      }
    }
    # then add the legend
    legend("topright", legendlabel, cex=0.5, col=legendcolor, lwd=legendwidth, lty=legendpattern)
    
  }
  
  #RETURNING RESULTS
  results=list() 
  results$N=N
  results$predictors=data.frame(cbind(predictors,harm,probability))
  names(results$predictors)=c("predictor","harm.applied","probability")
  results$interventions.avoided.per=interventionper
  results$net.benefit=nb
  results$interventions.avoided=interv
  
  return(results)
  
}  
dca(data=data, outcome="CT.CACscore_above400", predictors=c("LR","SVM","RF","XGB","MLP"), smooth="TRUE")



#Distributions of theprediction probabilities clustered by the presence of severe calcificarion
library(showtext)
showtext_auto(enable = TRUE)
newdata <- data[order(data$SVM),]
newdata
nrow(newdata)
newdata$number<-c(1:1157)
library(ggplot2)
newdata$CT.CACscore_above400 <- factor(newdata$CT.CACscore_above400, levels=c("1", "0"))
p1 <- ggplot(data=newdata, aes(x=number, y=SVM)) + 
  geom_point(aes(color=CT.CACscore_above400)) +
  scale_colour_manual(values=c("indianred2", "mediumaquamarine")) +
  theme_light() +
  geom_vline(xintercept=289, col="grey25", lty=3) +
  geom_vline(xintercept=578, col="grey25", lty=3) +
  geom_vline(xintercept=867, col="grey25", lty=3) +
  labs(x = "Patients sorted in order of risk", y = "Predicted risk of severe CAC") +
  guides(color=guide_legend(override.aes = list(size=2))) +
  theme(legend.position="none",
        axis.text.x = element_text(size=14))  
p1





#Part_3
library(xlsx)
library(survival)
library(compareC)
library(survIDINRI)
data<-read.xlsx("D:/Data_prognosis.xlsx",1)
#Cox proportional hazards regression analysis of ML-CAC score in different models
#For the primary end points:
#Crude model
fit_crude_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_crude_Q234) 
fit_crude_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                               + ML_CAC_quartiles, data = data)
summary(fit_crude_quartiles) 
#Model1
fit_model1_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + Age + Gender_female
                         + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model1_Q234) 
fit_model1_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + Age + Gender_female 
                              + ML_CAC_quartiles, data = data)
summary(fit_model1_quartiles) 
#Model2
fit_model2_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + Age + Gender_female 
                         + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                         + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model2_Q234) 
fit_model2_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + Age + Gender_female
                              + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                              + ML_CAC_quartiles, data = data)
summary(fit_model2_quartiles) 
#Model3
fit_model3_Q234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                           + Age + Gender_female
                         + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                         + Revascularization
                         + ML_CAC_Q2 + ML_CAC_Q3 + ML_CAC_Q4, data = data)
summary(fit_model3_Q234) 
fit_model3_quartiles <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + Age + Gender_female 
                              + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                              + Revascularization
                              + ML_CAC_quartiles, data = data)
summary(fit_model3_quartiles) 




#Cox proportional hazards regression analysis of CT_CAC score in different models:
#For the primary end point:
#Crude model
fit_crude_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                         + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_crude_234) 
fit_crude_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                + CT_CAC_categories, data = data)
summary(fit_crude_categories) 
#Model1
fit_model1_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + Age + Gender_female
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model1_234) 
fit_model1_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + Age + Gender_female 
                               + CT_CAC_categories, data = data)
summary(fit_model1_categories) 
#Model2
fit_model2_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + Age + Gender_female
                        + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model2_234) 
fit_model2_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + Age + Gender_female
                               + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                               + CT_CAC_categories, data = data)
summary(fit_model2_categories) 
#Model3
fit_model3_234 <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                          + Age + Gender_female
                        + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                        + Revascularization
                        + CT_CAC_2 + CT_CAC_3 + CT_CAC_4, data = data)
summary(fit_model3_234) 
fit_model3_categories <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                                 + Age + Gender_female 
                               + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                               + Revascularization
                               + CT_CAC_categories, data = data)
summary(fit_model3_categories) 
#For the second primary end point, all the codes remain unchanged except for the time and status. 



#Prognostic analysis for the improvement of C-index, NRI, and IDI
#For primary end points:
#compared ML-CAC scores performance with CT-CAC scores 
fit_ML_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                      ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4, data = data)
fit_CT_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~ 
                      CT_CAC_2+ CT_CAC_3 + CT_CAC_4, data = data)
data$ML_CAC <- fit_ML_CAC$linear.predictors 
data$CT_CAC <- fit_CT_CAC$linear.predictors 
compareC(data$primary_end_point_time, data$primary_end_point, -data$ML_CAC, -data$CT_CAC)
#∆C-index over a basic traditional Cox model
fit_basic_model <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                           + Age + Gender_female + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR, data = data)
fit_basic_model_ML_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                                  + Age + Gender_female + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                                + ML_CAC_Q2+ ML_CAC_Q3 + ML_CAC_Q4, data = data)
fit_basic_model_CT_CAC <- coxph(Surv(primary_end_point_time, primary_end_point) ~
                                  + Age + Gender_female + SmokingHistory + Hypertension + Diabetes + SystolicBP + Glucose + LDLCholesterol + eGFR
                                + CT_CAC_2+ CT_CAC_3 + CT_CAC_4, data = data)
data$basic_model <- fit_basic_model$linear.predictors 
data$basic_model_ML_CAC  <- fit_basic_model_ML_CAC$linear.predictors 
data$basic_model_CT_CAC  <- fit_basic_model_CT_CAC$linear.predictors 
compareC(data$primary_end_point_time, data$primary_end_point, -data$basic_model, -data$basic_model_ML_CAC)  #ML_CAC
compareC(data$primary_end_point_time, data$primary_end_point, -data$basic_model, -data$basic_model_CT_CAC)  #CT_CAC
#Improvement in the continuous NRI amd IDI over a basic traditional Cox model
indata0=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,Age,Gender_female,SmokingHistory,Hypertension,Diabetes,SystolicBP,Glucose,LDLCholesterol,eGFR)))
indata1=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,Age,Gender_female,SmokingHistory,Hypertension,Diabetes,SystolicBP,Glucose,LDLCholesterol,eGFR,
                                        ML_CAC_Q2, ML_CAC_Q3, ML_CAC_Q4)))
indata2=as.matrix(subset(data, select=c(primary_end_point_time, primary_end_point,Age,Gender_female,SmokingHistory,Hypertension,Diabetes,SystolicBP,Glucose,LDLCholesterol,eGFR,
                                        CT_CAC_2, CT_CAC_3, CT_CAC_4)))                                        
covs0<-as.matrix(indata0[,c(-1,-2)])
covs1<-as.matrix(indata1[,c(-1,-2)])
covs2<-as.matrix(indata2[,c(-1,-2)])
set.seed(1234)
x1 <- IDI.INF(indata0[ ,1:2], covs0, covs1, 36,  npert=1000) #for ML_CAC 
IDI.INF.OUT(x1) #for ML_CAC
x2 <- IDI.INF(indata0[ ,1:2], covs0, covs2, 36,  npert=1000) #for CT_CAC 
IDI.INF.OUT(x2) #for CT_CAC
#For the secondary end point, all the codes remain unchanged except for the time and status. 




