
library(MASS)
Bank_data <- read.csv(file.choose())
#("C:/RAVI/Data science/Assignments/Module 9 Logistic regression/LR Assignment dataset2/bank_data.csv/bank_data.csv")

View(Bank_data)

summary(Bank_data)
attach(Bank_data)

sum(is.na(Bank_data))

# GLM function use sigmoid curve to produce desirable results 
# The output of sigmoid function lies in between 0-1
#(GLM)generalised linear model



model <- glm(y ~ ., data = Bank_data, family = "binomial")
summary(model)

# To calculate the odds ratio manually we going r going to take exp of coef(model)
exp(coef(model))

# Confusion matrix table 
prob <- predict(model,Bank_data,type="response")
prob
# We are going to use NULL and Residual Deviance to compare the between different models

# Confusion matrix and considering the threshold value as 0.5 
confusion <- table(prob > 0.5, Bank_data$y)
confusion

table(prob>0.5)


# Model Accuracy 
Accuracy <- sum(diag(confusion)/sum(confusion))
Accuracy 


# Creating empty vectors to store predicted classes based on threshold value
pred_values <- NULL
yes_no <- NULL

pred_values <- ifelse(prob > 0.5, 1, 0)
yes_no <- ifelse(prob > 0.5,"yes","no")

# Creating new column to store the above values
Bank_data[ , "prob"] <- prob
Bank_data[ , "pred_values"] <- pred_values
Bank_data[ , "yes_no"] <- yes_no

View(Bank_data[ , c(32:34)])

table(Bank_data$y, Bank_data$pred_values)

# Calculate the below metrics
# precision | recall | True Positive Rate | False Positive Rate | Specificity | Sensitivity

Specificity=39013/(39013+909)=0.977 #TN/(TN+FP) OR True_Negetive_Rate
Sensitivity <-1702/(1702+3587)=0.32  #TP/(TP+FN) OR True_Positive_Rate
False_Positive_Rate <- 1-Specificity=0.02

#############

# ROC Curve => used to evaluate the betterness of the logistic model
# more area under ROC curve better is the model 
# We will use ROC curve for any classification technique not only for logistic

install.packages("ROCR")
library(ROCR)
rocrpred <- prediction(prob, Bank_data$y)
rocrperf<-performance(rocrpred,'tpr','fpr')

str(rocrperf)

plot(rocrperf)

plot(rocrperf,colorize=T,text.adj=c(-0.2,1.7))
# More area under the ROC Curve better is the logistic regression model obtained

str(rocrperf)
rocr_cutoff <- data.frame(cut_off = rocrperf@alpha.values[[1]],fpr=rocrperf@x.values,tpr=rocrperf@y.values)
colnames(rocr_cutoff) <- c("cut_off","FPR","TPR")
View(rocr_cutoff)

library(dplyr)

rocr_cutoff$cut_off <- round(rocr_cutoff$cut_off,6)

# Sorting data frame with respect to tpr in decreasing order 
rocr_cutoff <- arrange(rocr_cutoff,desc(TPR))
View(rocr_cutoff)

library(ROCR)
auc(rocrperf)
