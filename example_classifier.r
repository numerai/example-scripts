# load randomForest Library
require(randomForest)
# load the training data
train <- read.csv("numerai_training_data.csv", head=T)
#set the seed for reproducibility
set.seed(98415)
# fit a randomForest model
fit <- randomForest(as.factor(target) ~ ., data=train, importance=TRUE, do.trace = TRUE, ntree=50)
#plot the variable importance
varImpPlot(fit)
#load the test data
test <- read.csv("numerai_tournament_data.csv", head=T)
Prediction <- predict(fit, test[,-1],type="prob")
test$probability <- Prediction[,"1"]
pred <- test[,c("t_id","probability")]
#write the prediction file for submission
write.csv(pred, file="example_pred.csv", quote=F, row.names=F)
