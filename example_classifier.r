#!/usr/bin/env Rscript

# Example classifier on Numerai data using random forest classifier.
# To get started, install the required packages: install.packages(randomForest, cvTools)

require(randomForest)

MultiLogLoss <- function(act, pred){
  eps <- 1e-15
  pred <- pmin(pmax(pred, eps), 1 - eps)
  sum(act * log(pred) + (1 - act) * log(1 - pred)) * -1/NROW(act)
}

# Set seed for reproducibility
set.seed(0)

print("Loading data...")
train <- read.csv("numerai_training_data.csv", head=T)
test <- read.csv("numerai_tournament_data.csv", head=T)

print("Testing...")
folds <- cut(seq(1, nrow(train)), breaks=3, labels=FALSE)
total <- 0.0
for(i in 1:3) {
    testIndexes <- which(folds==i, arr.ind=TRUE)
    testData <- train[testIndexes, ]
    trainData <- train[-testIndexes, ]
    fit <- randomForest(as.factor(target) ~., data=trainData, importance=FALSE, do.trace=TRUE, ntree=50)
    prediction <- predict(fit, testData, type="prob")
    loss <- MultiLogLoss(testData["target"], prediction)
    print(loss)
    total <- total + loss
}
sprintf("Mean cross validation log loss: %f", total/3)

print("Training...")
fit <- randomForest(as.factor(target) ~ ., data=train, importance=FALSE, do.trace=TRUE, ntree=50)

print("Predicting...")
Prediction <- predict(fit, test[,-1], type="prob")
test$probability <- Prediction[,"1"]
pred <- test[,c("t_id", "probability")]

print("Writing predictions to submission.csv")
write.csv(pred, file="submission.csv", quote=F, row.names=F)
