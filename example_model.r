#Tutorial and Tips for Numerai
library(Metrics)
library(gbm)
setwd("~/Downloads/numerai_datasets/")
#the training data is used to train your model how to predict the targets
train<-read.csv("numerai_training_data.csv", header=T)
#the tournament data is the data that Numerai uses to evaluate your model
tournament<-read.csv("numerai_tournament_data.csv", header=T)
#the tournament data contains validation data, test data and live data
#validation is used to test your model locally so we separate that
validation<-tournament[tournament$data_type=="validation",]

#There are multiple targets in the training data which you can choose to model using the features.
#Numerai does not say what the features mean but that's fine; we can still build a model.
#Here we select the bernie_target
train_bernie<-subset(train, select=-c(id, era, data_type, target_charles, target_elizabeth, target_jordan, target_ken, target_frank, target_hillary))
head(train_bernie)

#build a gbm model to predict this target
model<-gbm(target_bernie~., data=train_bernie, n.trees=10, shrinkage=0.01, interaction.depth=2, train.fraction=1, bag.fraction=0.5, verbose=T)
#look at the relative importance of the features (you can see the model relies strongly on certain features)
head(summary(model))

#based on the model we can predict the probability of each row being a bernie_target in the validation data
probabilities<-predict.gbm(model, validation, n.trees=10, type="response")
head(probabilities)

#we can see the probability does seem to be good at predicting the true target correctly
validation$target_bernie[1:6]
round(probabilities[1:6])

#but overall the accuracy is very low
sum(round(probabilities)==validation$target_bernie)/nrow(validation)

#the targets for each of the tournaments are very correlated
cor(validation$target_bernie, validation$target_elizabeth)
#you can see that target_elizabeth is accurate using the bernie model as well
sum(round(probabilities)==validation$target_elizabeth)/nrow(validation)

#Numerai measures models on AUC. The higher the AUC the better.
#Numerai only pays models with AUC that beat the benchmark on the live portion of the tournament data.
#Our validation AUC isn't very good.
auc(validation$target_bernie, probabilities)

#to submit predictions from your model to Numerai, predict on the entire tournament data
tournament$probability_bernie<-predict.gbm(model, tournament, n.trees=10, type="response")

#create your submission
submission<-subset(tournament, select=c(id, probability_bernie))
head(submission)

#save your submission and now upload it to Numerai on https://numer.ai
write.csv(submission, file="bernie_submission2.csv", row.names=F)


#TIPS TO IMPROVE YOUR MODEL

#1. Use eras
#In this example, we dropped era column but you can use the era column to improve peformance across eras
#You can take a model like the above and use it to generate probabilities on the training data, and
#look at the the eras where your model was <0.693 and then build a new model on those bad eras to
#combine with your main model. In this way, you may be hedged to the risk of bad eras in the future.
#Advanced tip: To take this further, you could add the objective of doing consistenty well across all eras
#directly into the objective function of your machine learning model.

#2. Use feature importance
#As per above, you don't want your model to rely too much on any particular type of era. Similarly, you
#don't want your model to rely too much on any particular type of feature. If your model relies heavily on
#one feature (in linear regression, some feature has very high coefficient), then if that feature doesn't work
#in a particular era then your model will perform poorly. If your model balances its use of features then it is
#more likely to be consistent across eras.

#3. Use all the targets
#As we saw above, a model trained on one target like target_bernie might be good at predicting another target
#like target_elizabeth. Blending models built on each target could also improve your AUC.
