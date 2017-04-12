#!/usr/bin/env Rscript

# Example classifier on Numerai data using logistic regression classifier.

# Set seed for reproducibility
set.seed(0)

print("Loading data...")
# Load the data from the CSV files
train <- read.csv("training_data.csv", head=T)
test <- read.csv("tournament_data.csv", head=T)

print("Removing all columns from train other than features and target...")
train <- train[,grep("feature|target",names(train))]
print("Selecting test and live rows and Removing all columns from test other than features and id...")
test  <- test[test$data_type %in% c("test","live"),grep("id|feature",names(test))]

print("Training...")
# This is your model that will learn to predict. Your model is trained on the numerai_training_data
model <- glm(as.factor(target) ~ ., data=train, family=binomial(link='logit'))

print("Predicting...")
# Your trained model is now used to make predictions on the numerai_tournament_data
predictions <- predict(model, test[,-1], type="response")
test$probability <- predictions
pred <- test[,c("id", "prediction")]

print("Writing predictions to predictions.csv")
# Save the predictions out to a CSV file
write.csv(pred, file="predictions.csv", quote=F, row.names=F)
# Now you can upload your predictions on numer.ai
