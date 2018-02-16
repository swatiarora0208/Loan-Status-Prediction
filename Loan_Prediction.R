# Setting working directory
setwd("G:/projects/AnalyticsVidhya_LoanPredictionProblem")

#Loading the required libraries
install.packages('caret')
library('caret')
install.packages('dplyr')
require(dplyr)
install.packages('Amelia')
library(Amelia)
install.packages('ggplot2')
library(ggplot2)

#Loading the dataset
loan<-read.csv('train.csv',na.strings=c("","NA"))
loan.test<-read.csv('test.csv',na.strings=c("","NA"))

#Let's see if the structure of dataset data
str(loan)
str(loan.test)
head(loan)


#Creating a function to calculate number of missing values, proportion of missing values 
#and number of unique values across each column
missing_values = function(input)
{
  n = length(colnames(input)) # number of columns
  a <- NULL
  b <- NULL
  c <- NULL
  for(i in 1:n) 
  {
    a[i]=sum(is.na(input[,i])) 
    b=a/nrow(input) 
    c[i]=length(unique(input[,i])) 
  }
  result=data.frame(colnames(input),a,b,c) 
  colnames(result) = c("column Name", "# Missing Values", "% Missing Value", "Unique Values")
  return(result) 
}

##Applying the missing_values function on train
missing_train<-missing_values(loan)
View(missing_train)
missing_test <- missing_values(loan.test)
View(missing_test)


# Visualizing the missing values
missmap(loan,y.at=c(1),y.labels = c(''),col=c('yellow','black'),legend = FALSE)
missmap(loan.test,y.at=c(1),y.labels = c(''),col=c('yellow','black'),legend = FALSE)
sapply(loan, function(x) sum(is.na(x)))
View(loan)
plot(loan$Gender,loan$LoanAmount)
plot(loan$Gender,loan$Loan_Status)

# finding the correlation between data
install.packages('psych')
library(psych)
cor.ci(loan[7:11],method = 'spearman')

#Impute Data for missing values
#install MICE
#the methods used by this package are:
#PMM (Predictive Mean Matching)  - For numeric variables
#logreg(Logistic Regression) - For Binary Variables( with 2 levels)
#polyreg(Bayesian polytomous regression) - For Factor Variables (>= 2 levels)
#Proportional odds model (ordered, >= 2 levels)
install.packages("mice")
library(mice)
md.pattern(loan)
#?mice
loan.numerial <- subset(loan,select = c(LoanAmount,Loan_Amount_Term,Credit_History)) 
loan.categorical <- subset(loan,select = c(Gender,Married,Dependents,Self_Employed))
imputed_num_Data <- mice(loan.numerial, m=5, maxit = 5, method = 'pmm')
imputed_cat_Data <- mice(loan.categorical, m=5, maxit = 5, method = 'polyreg')
i = complete(imputed_num_Data,1)
j = complete(imputed_cat_Data,1)
str(j)
combined_loan <- cbind(i,j)
combined_loan <- cbind(combined_loan,loan[c('Education','ApplicantIncome','CoapplicantIncome','Property_Area','Loan_Status')])
missmap(combined_loan,y.at=c(1),y.labels = c(''),col=c('yellow','black'),legend = FALSE)
sum(is.na(combined_loan))
str(combined_loan)

# imputing data for test set
sapply(loan.test, function(x) sum(is.na(x)))

loan.test.numerial <- subset(loan.test,select = c(LoanAmount,Loan_Amount_Term,Credit_History)) 
loan.test.categorical <- subset(loan.test,select = c(Gender,Dependents,Self_Employed))
test_imputed_num_Data <- mice(loan.test.numerial, m=5, maxit = 5, method = 'pmm')
test_imputed_cat_Data <- mice(loan.test.categorical, m=5, maxit = 5, method = 'polyreg')
i = complete(test_imputed_num_Data,1)
j = complete(test_imputed_cat_Data,1)
combined_test_loan <- cbind(i,j)
combined_test_loan <- cbind(combined_test_loan,loan.test[c('Married','Education','ApplicantIncome','CoapplicantIncome','Property_Area')])
missmap(combined_test_loan,y.at=c(1),y.labels = c(''),col=c('yellow','black'),legend = FALSE)
sum(is.na(combined_test_loan))
str(combined_loan)
X_test <- combined_test_loan

# convert Credit History from int to factor
combined_loan$Credit_History <- as.factor(combined_loan$Credit_History)
combined_test_loan$Credit_History <- as.factor(combined_test_loan$Credit_History)


#Spliting training set into two parts based on outcome: 75% and 25%

index <- createDataPartition(combined_loan$Loan_Status, p=0.75, list=FALSE)
trainSet <- combined_loan[ index,]
testSet <- combined_loan[-index,]

# checking p-values of all predictors using logistic regression
glm.fit=glm(Loan_Status~.,data=trainSet,family=binomial)
summary(glm.fit)
glm.probs=predict(glm.fit,newdata=testSet,type="response") 
#glm.pred=ifelse(glm.probs >0.5,"Up","Down")


#Defining the training controls for multiple models
fitControl <- trainControl(
  method = "cv",
  number = 5,
  savePredictions = 'final',
  classProbs = T)


#Defining the predictors and outcome
predictors<-c("Credit_History", "LoanAmount", "Loan_Amount_Term", "ApplicantIncome",
              "CoapplicantIncome","Property_Area")
predictors<-c("Credit_History","Property_Area", 'Married')
outcomeName<-'Loan_Status'

#------------ Random Forest ----------------#
#Training the random forest model
model_rf<-train(trainSet[,predictors],trainSet[,outcomeName],method='rf',trControl=fitControl,tuneLength=3)

#Predicting using random forest model
testSet$pred_rf<-predict(object = model_rf,testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,testSet$pred_rf)

#------------ Logistic Regression ----------------#
#Training the Logistic regression model
model_lr<-train(trainSet[,predictors],trainSet[,outcomeName],method='glm',trControl=fitControl,tuneLength=3)

#Predicting using knn model
testSet$pred_lr<-predict(object = model_lr,testSet[,predictors])

#Checking the accuracy of the random forest model
confusionMatrix(testSet$Loan_Status,testSet$pred_lr)

#------------ Gradient Boosting model ----------------#
#Training the model
model_gbm<-train(trainSet[,predictors],trainSet[,outcomeName],method='gbm',trControl=fitControl,tuneLength=3)

#Predicting using gbm model
testSet$pred_gbm<-predict(object = model_gbm,testSet[,predictors])

#Checking the accuracy of the gradient boosting model
confusionMatrix(testSet$Loan_Status,testSet$pred_gbm)

#------------ knn model ----------------#
#Training the model
model_knn<-train(trainSet[,predictors],trainSet[,outcomeName],method='knn',trControl=fitControl,tuneLength=3)

#Predicting using knn model
testSet$pred_knn<-predict(object = model_knn,testSet[,predictors])

#Checking the accuracy of the knn model
confusionMatrix(testSet$Loan_Status,testSet$pred_knn)

#------------ Naive Bayes model ----------------#
library("klaR")
#Training the model
model_nb<-train(trainSet[,predictors],trainSet[,outcomeName],method='nb',trControl=fitControl,tuneLength=3)

#Predicting using gbm model
testSet$pred_nb<-predict(object = model_nb,testSet[,predictors])

#Checking the accuracy of the gradient boosting model
confusionMatrix(testSet$Loan_Status,testSet$pred_nb)

#---------- Predicting the values for test dataset using logistic regression: 77% ---------------#

loan.test$Loan_Status<-predict(object = model_lr,combined_test_loan[,predictors])
View(loan.test)
# save the file as loan_id and loan_status to 
write.csv(loan.test[,c('Loan_ID','Loan_Status')], file = "Sample_Submission.csv",row.names=TRUE)


#---------- Predicting the values for test dataset using random forest : 77.77% ---------------#

loan.test$Loan_Status_rf<-predict(object = model_rf,combined_test_loan[,predictors])
View(loan.test)
# save the file as loan_id and loan_status to 
write.csv(loan.test[,c('Loan_ID','Loan_Status_rf')], file = "Sample_Submission.csv",row.names=TRUE)

