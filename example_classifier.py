import pandas as pd
import numpy as np
from sklearn import ensemble, metrics


print "Loading data"
training_data = pd.read_csv('numerai_training_data.csv', header=0)
prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

Y = training_data['target']
X = training_data.drop('target', axis=1)

t_id = prediction_data['t_id']
X_prediction = prediction_data.drop('t_id', axis=1)

print "Training"
rf_classifer = ensemble.RandomForestClassifier(n_estimators = 100, n_jobs=-1)

# First run the model on 80% of the data to get an idea how well our model is
# doing.
cross_validation = np.random.rand(len(X)) < 0.8
rf_classifer.fit(X[cross_validation],Y[cross_validation])
Y_test = rf_classifer.predict_proba(X[~cross_validation])
test_logloss = metrics.log_loss(Y[~cross_validation], Y_test)
print "Test logloss: {}".format(test_logloss)

rf_classifer.fit(X,Y)
Y_prediction = rf_classifer.predict_proba(X_prediction)

print "Predicting"
# The random forest classifer returns two columns,
# [probability of 0, probability of 1]
# We are just interested in the probability that the target is 1.
results = Y_prediction[:,1]
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(t_id).join(results_df)

print "Writing out to submission.csv"
joined.to_csv('submission.csv', index = False)
