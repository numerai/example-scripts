#Tutorial and Tips for Numerai

#***
#The stock market is the hardest data science problem in the world.
#If it seems easy, you're doing it wrong.
#We don't know how to solve this problem. But here is what we know.
#We need your help. You take it from here.
#***
library(Metrics)
library(gbm)
library(randomForest)

# for automatic download of data and upload of submissions have a look at the package Rnumerai
# for the latest development release use devtools::install_github("Omni-Analytics-Group/Rnumerai") or pak::pkg_install("Omni-Analytics-Group/Rnumerai")
# for the CRAN version use install.packages("Rnumerai")

print("THIS SCRIPT DOWNSAMPLES DATA AND WILL LIKELY UNDERPERFORM THE example_predictions.csv")

#the training data is used to train your model how to predict the targets
training_data<-read.csv("numerai_training_data.csv", header=T)
#the tournament data is the data that Numerai uses to evaluate your model
tournament<-read.csv("numerai_tournament_data.csv", header=T)
#the tournament data contains validation data, test data and live data
#validation is a hold out set out of sample and further in time (higher eras) than the training data
validation<-tournament[tournament$data_type=="validation",]

benchmark<-0.002
consistency<-function(data_with_probs, target_name){
  unique_eras<-unique(data_with_probs$era)
  era_correlation<-vector()
  i<-1
  while(i<=length(unique_eras)){
    this_data<-data_with_probs[data_with_probs$era==unique_eras[i],]
    era_correlation[i]<-cor(this_data[,names(this_data)==target_name], this_data$prediction, method="spearman")
    i<-i+1
  }
  consistency<-sum(era_correlation>benchmark)/length(era_correlation)
  consistency
}

#the training data is large so we reduce it by 10x in order to speed up this script
#NB reducing the training data like this is not recommended for a production ready model
nrow(training_data)
set.seed(10)
#remove the following line to not downsample. Note: it will take much longer to run
sample_rows<-sample(1:nrow(training_data), ceiling(nrow(training_data)/10))
train<-training_data[sample_rows,]

#we remove the id, era, and data_type columns for now
train<-train[,-c(1:3)]
nrow(train)

#there are a large number of features
ncol(train)
#features are a number of feature groups of different sizes (intelligence, charisma, strength, dexterity, constitution, wisdom)
#Numerai does not disclose what the features or feature groups mean; they are abstract and obfuscated
#however, features within feature groups tend to be related in some way
#models which have large feature importance in one group tend to do badly for long stretches of time (eras).
head(names(train),20)

#the target variable has 5 ordinal outcomes (0,0.25,0.5,0.75,1)
#we train a model on the features to predict this target
summary(train$target)

#on this dataset, feature importance analysis is very important
#we build a random forest to understand which features tend to improve the model out of bag
#because stocks within eras are not independent, we use small bag sizes (sampsize=10%) so the out of bag estimate is meaningful
#we use a small subset of features for each tree to build a more feature balanced forest (mtry=10%)
set.seed(10)
forest<-randomForest(target~., data=train, ntree=50, mtry=ceiling(0.1*ncol(train)-1), sampsize=ceiling(0.1*nrow(train)), importance=T, maxnodes=5)
#a good model might drop the bad features according to the forest before training a final model
#if a feature group or feature is too good, it might also be a good idea to drop to improve the feature balance and improve consistency of the model
imp<-importance(forest)
imp<-imp[order(imp[,1], decreasing = FALSE),]
head(imp)

#based on the forest we can generate predictions on the validation set
#the validation set contains eras further in time than the training set
#the validation set is not "representative of the live data"; no one knows how the live data will behave
#usually train performance < validation performance < live performance because live data is many eras into the future from the training data
predictions<-predict(forest, validation)
head(predictions)
#Numerai measures performance based on Rank Correlation between your predictions and the true targets
cor(validation$target, predictions, method="spearman")
val<-validation
val$prediction<-predictions
#consistency is the fraction of months where the model achieves better correlation with the targets than the benchmark
consistency(val, "target")

#we try a gbm model; we also choose a low bag fraction of 10% as a strategy to deal with within-era non-independence (which is a property of this data)
#(if you take a sample from one era and a different sample from the same era that sample is not really out of sample because the observations occured in the same era)
#having small bags also improves the out of bag estimate for the optimal number of trees
set.seed(10)
model<-gbm(target~., data=train, n.trees=50, shrinkage=0.01, interaction.depth=5, train.fraction=1, bag.fraction=0.1, verbose=T)
#looking at the relative importance of the features we can see the model relies more on some features than others
head(summary(model))
best.iter <- gbm.perf(model, method="OOB")
best.iter

#we check performance of the gbm model on the out of sample validation data
predictions<-predict.gbm(model, validation, n.trees=best.iter, type="response")
head(predictions)
cor(validation$target, predictions, method="spearman")
val<-validation
val$prediction<-predictions
consistency(val, "target")

#the gbm model and random forest model have the same consistency (number of eras where correlation > benchmark) even though correlation are different
#improving consistency can be more important than improving standard machine learning metrics like RMSE
#good models might train in such a way as to minimize the error across eras (improve consistency) not just reduce the error on each training example

#so far we have ignored eras for training
#eras are in order of time; for example, era4 is before era5
#the reason to use eras is that cross validation on rows will tend to overfit (rows within eras are not independent)
#so it's much better to cross validate within eras for example: take a subset of eras, build a model and test on the out of sample subset of eras
#this will give a better estimate of how well your model will generalize
#the validation set is not special; it is just an out of sample set of eras greater than the training set
#some users might choose to train on the validation set as well

#to give you a sense of how to use eras we train a model on the first half of the eras and test it on the second half
#we take another 10x smaller sample of data to speed up the script
ordered_eras<-unique(training_data$era)
train<-training_data[sample_rows,]
first_half<-train[train$era%in%ordered_eras[1:ceiling(length(ordered_eras)/2)],]
second_half<-train[!(train$id%in%first_half$id),]

#we remove id, era, data_type column and train a gbm model on the first half of the data
set.seed(10)
model<-gbm(target~., data=first_half[,-c(1:3)], n.trees=100, shrinkage=0.01, interaction.depth=5, train.fraction=1, bag.fraction=0.1, verbose=T)
best.iter <- gbm.perf(model, method="OOB")
best.iter

predictions<-predict.gbm(model, second_half, n.trees=best.iter, type="response")
#our correlation score is good; what we appeared to learn generalizes well on the second half of the eras
0.5+cor(second_half$target, predictions, method="spearman")
sec<-second_half
sec$prediction<-predictions
#consistency is the fraction of months where the model achieves better correlation with the targets than the benchmark
consistency(sec, "target")

#but now we try build a model on the second half of the eras and predict on the first half
set.seed(10)
model<-gbm(target~., data=second_half[,-c(1:3)], n.trees=100, shrinkage=0.01, interaction.depth=5, train.fraction=1, bag.fraction=0.1, verbose=T)
best.iter <- gbm.perf(model, method="OOB")
best.iter

predictions<-predict.gbm(model, first_half, n.trees=best.iter, type="response")
#our correlation score is surprisingly low and surprisingly different from when we trained on the first half
#this means our model is not very consistent, and it's possible that we will see unexpectedly low performance on the test set or live set
#it also shows that our validation performance is likely to greatly overestimate our performance--this era-wise cross validation is more valuable
#a model whose performance when training on the first half is the same as training on the second half would likely be more consistent
#and would likely perform better on the tournament data and the live data
cor(first_half$target, predictions, method="spearman")
fir<-first_half
fir$prediction<-predictions
#consistency is the fraction of months where the model achieves better correlation with the targets than the benchmark
consistency(fir, "target")

#Numerai only pays models with correlations that beat the benchmark on the live portion of the tournament data
#to submit predictions from your model to Numerai, predict on the entire tournament data
#we choose use our original forest model for our final submission
tournament$prediction<-predict(forest, tournament)

#create your submission
submission<-subset(tournament, select=c(id, prediction))
head(submission)

#save your submission and now upload it to https://numer.ai
write.csv(submission, file="submission.csv", row.names=F)
