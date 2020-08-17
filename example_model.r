#Tutorial and Tips for Numerai

#***
#The stock market is the hardest data science problem in the world.
#If it seems easy, you're doing it wrong.
#We don't know how to solve this problem. But here is what we know.
#We need your help. You take it from here.
#***

library(magrittr)
library(tidyverse)
library(ranger)
library(gbm)

set.seed(1337)

#function to calculate them correlation between your model's predictions and the target across the validation eras
corrfun <- function(df) {
  correlation_data <- df %>%
    group_by(era) %>%
    summarise(r_model = cor(target_kazutsugi, predictions, method = "spearman")) %>%
    ungroup()
  
}

print(
  "This script uses down sampling and random forest and will likely underperform the example_predictions.csv"
)

#the training data is used to train your model how to predict the targets
training_data <- read_csv("numerai_training_data.csv")

#the tournament data is the data that Numerai uses to evaluate your model
tournament_data <- read_csv("numerai_tournament_data.csv")

#the tournament data contains validation data, test data and live data
#validation is a hold out set out of sample and further in time (higher eras) than the training data
#let's separate the validation data from the tournament data
validation <- tournament_data %>%
  filter(data_type == "validation")

#we will not be using id and data_type when training the model so these columns should be removed from the training data
training_data %<>%
  select(-id,-data_type)

#the training data is large so we reduce it by 10x in order to speed up this script
#note that reducing the training data like this is not recommended for a production ready model
nrow(training_data)

#remove the following lines to skip the down sampling. Note: training will take much longer to run
training_data_downsampled <- training_data %>%
  sample_frac(0.1)

#there are a large number of features
training_data_downsampled %>%
  select(-era, -target_kazutsugi) %>%
  colnames() %>%
  length()

#features are a number of feature groups of different sizes (intelligence, charisma, strength, dexterity, constitution, wisdom)
#Numerai does not disclose what the features or feature groups mean; they are abstract and obfuscated
#however, features within feature groups tend to be related in some way
#models which have large feature importance in one group tend to do badly for long stretches of time (eras)
training_data_downsampled %>%
  colnames() %>%
  head(20)

#the target variable has 5 ordinal outcomes (0, 0.25, 0.5, 0.75, 1)
#we train a model on the features to predict this target
summary(training_data_downsampled$target_kazutsugi)

#on this dataset, feature importance analysis is very important
#we build a random forest to understand which features tend to improve the model out of bag
#because stocks within eras are not independent, we use small bag sizes (sample size = 10%) so the out of bag estimate is meaningful
#we use a small subset of features for each tree to build a more feature balanced forest (mtry = floor(num_features))
#we use all features except for era
forest <-
  ranger(
    target_kazutsugi ~ .,
    data = training_data_downsampled[, -1],
    num.trees  = 500,
    max.depth = 5,
    sample.fraction = 0.1,
    importance = "impurity"
  )

feature_importance <- as.data.frame(importance(forest))
colnames(feature_importance) <- "feature_importance"

#a good model might drop the bad features according to the forest before training a final model
#if a feature group or feature is too good, it might also be a good idea to drop it to improve the feature balance and improve consistency of the model

#let us display the, according to our model, 20 most important features on this subset of the training data
feature_importance %>%
  arrange(-feature_importance) %>%
  head(20)


#based on the forest we can generate predictions on the validation set
#the validation set contains eras further in time than the training set
#the validation set is not "representative of the live data"; no one knows how the live data will behave
#usually train performance < validation performance < live performance because live data is many eras into the future from the training data
predictions <- predict(forest, validation)
head(predictions$predictions)

#Numerai measures performance based on rank correlation between your predictions and the true targets
cor(validation$target_kazutsugi,
    predictions$predictions,
    method = "spearman")

#a correlation of 0.017 might be on the lower end, this model could perhaps use some extra tuning or more training data
validation$predictions <- predictions$predictions

#let us look at the era to era variation when it comes to correlation
correlations <- corrfun(validation)
correlations %>% print(n = Inf)

#era 199 seems difficult, our model has a correlation of -0.0384 on that particular era
#successful models typically have few eras with large negative correlations

#consistency is the fraction of months where the model achieves better than zero correlation with the targets
sum(ifelse(correlations$r_model > 0, 1, 0)) / nrow(correlations)

#the Sharpe ratio is defined as the average era correlation divided by their standard deviation

mean(correlations$r_model) / sd(correlations$r_model)

#we can try a GBM model; we also choose a low bag fraction of 10% as a strategy to deal with within-era non-independence (which is a property of this data)
#(if you take a sample from one era and a different sample from the same era that sample is not really out of sample because the observations occured in the same era)
#having small bags also improves the out of bag estimate for the optimal number of trees
model <-
  gbm(
    target_kazutsugi ~ .,
    data = training_data_downsampled[,-1],
    n.trees = 50,
    shrinkage = 0.01,
    interaction.depth = 5,
    train.fraction = 1,
    bag.fraction = 0.1,
    verbose = T
  )

#looking at the relative importance of the features we can see the model relies more on some features than others
head(summary(model))
best.iter <- gbm.perf(model, method = "OOB")
best.iter

predictions <-
  predict.gbm(model, validation, n.trees = best.iter, type = "response")
head(predictions)
cor(validation$target_kazutsugi, predictions, method = "spearman")

validation$predictions <- predictions

#let us look at the era to era variation when it comes to correlation for our GBM model
correlations <- corrfun(validation)
correlations %>% print(n = Inf)

#let us calculate the consistency
sum(ifelse(correlations$r_model > 0, 1, 0)) / nrow(correlations)

#and the Sharpe ratio
mean(correlations$r_model) / sd(correlations$r_model)

#the GBM model and random forest model have the same consistency (number of eras where correlation is positive) even though correlations are different
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
training_data$era <- as.integer(as.factor(training_data$era))
first_half <- training_data %>%
  filter(era <= 60)
second_half <- training_data %>%
  filter(era > 60)

#we remove id, era, data_type column and train a GBM model on the first half of the data
model_first_half <-
  gbm(
    target_kazutsugi ~ .,
    data = first_half[, -1],
    n.trees = 50,
    shrinkage = 0.01,
    interaction.depth = 5,
    train.fraction = 1,
    bag.fraction = 0.1,
    verbose = T
  )
best.iter1 <- gbm.perf(model_first_half, method = "OOB")
best.iter1

predictions_second_half  <-
  predict.gbm(model_first_half,
              as.data.frame(second_half),
              n.trees = best.iter1,
              type = "response")

#our correlation score is good; what we appeared to learn generalizes well on the second half of the eras
cor(second_half$target_kazutsugi,
    predictions_second_half,
    method = "spearman")


second_half$predictions <- predictions_second_half

#let us investigate the correlations on the second half of the data
correlation_data_second_half <- second_half %>%
  group_by(era) %>%
  summarise(r_model = cor(target_kazutsugi, predictions, method = "spearman")) %>%
  ungroup()

second_half <- second_half %>% select(-predictions)

correlation_data_second_half %>% print(n = Inf)

#let us calculate the consistency on the second half of the data
sum(ifelse(correlation_data_second_half$r_model > 0, 1, 0)) / nrow(correlation_data_second_half)

#and the Sharpe ratio
mean(correlation_data_second_half$r_model) / sd(correlation_data_second_half$r_model)


#but now we try build a model on the second half of the eras and predict on the first half
model_second_half <-
  gbm(
    target_kazutsugi ~ .,
    data = second_half[, -1],
    n.trees = 50,
    shrinkage = 0.01,
    interaction.depth = 5,
    train.fraction = 1,
    bag.fraction = 0.1,
    verbose = T
  )
best.iter2 <- gbm.perf(model_second_half, method = "OOB")
best.iter

predictions_first_half <-
  predict.gbm(
    model_second_half,
    as.data.frame(first_half[, -1]),
    n.trees = best.iter,
    type = "response"
  )


#let us calculate the correlation
cor(first_half$target_kazutsugi, predictions_first_half,
    method = "spearman")


first_half$predictions <- predictions_first_half

#let us investigate the correlations on the second half of the data
correlation_data_first_half <- first_half %>%
  group_by(era) %>%
  summarise(r_model = cor(target_kazutsugi, predictions, method = "spearman")) %>%
  ungroup()

correlation_data_first_half %>% print(n = Inf)

#let us calculate the consistency on the second half of the data
sum(ifelse(correlation_data_first_half$r_model > 0, 1, 0)) / nrow(correlation_data_first_half)

#and the Sharpe ratio
mean(correlation_data_first_half$r_model) / sd(correlation_data_first_half$r_model)

#our correlation score is surprisingly low and surprisingly different from when we trained on the first half
#this means our model is not very consistent, and it's possible that we will see unexpectedly low performance on the test set or live set
#it also shows that our validation performance is likely to greatly overestimate our performance--this era-wise cross validation is more valuable
#a model whose performance when training on the first half is the same as training on the second half would likely be more consistent
#and would likely perform better on the tournament data and the live data

#remove unused data and models
rm(
  feature_importance,
  model,
  best.iter,
  best.iter1,
  best.iter2,
  model_first_half,
  model_second_half,
  first_half,
  second_half,
  training_data,
  validation,
  training_data_downsampled,
  predictions_first_half,
  predictions_second_half,
  correlation_data_first_half,
  correlation_data_second_half,
  correlations,
  predictions,
  prediction_kazutsugi,
)

#Numerai only pays models with correlations that beat the benchmark on the live portion of the tournament data
#to submit predictions from your model to Numerai, predict on the entire tournament data
#we choose use our original forest model for our final submission
prediction_kazutsugi <- predict(forest, tournament_data)
tournament_data$prediction_kazutsugi <-
  prediction_kazutsugi$predictions

#create your submission
submission <- tournament_data %>% select(id, prediction_kazutsugi)
head(submission)

#save your submission and now upload it to https://numer.ai
write_csv(submission, path = "kazutsugi_submission.csv")