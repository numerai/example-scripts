#!/usr/bin/env python
"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, linear_model


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("# Loading data...")
    # The training data is used to train your model how to predict the targets.
    train = pd.read_csv('numerai_training_data.csv', header=0)
    # The tournament data is the data that Numerai uses to evaluate your model.
    tournament = pd.read_csv('numerai_tournament_data.csv', header=0)

    # The tournament data contains validation data, test data and live data.
    # Validation is used to test your model locally so we separate that.
    validation = tournament[tournament['data_type'] == 'validation']

    # There are multiple targets in the training data which you can choose to model using the features.
    # Numerai does not say what the features mean but that's fine; we can still build a model.
    # Here we select the bernie_target.
    train_bernie = train.drop([
        'id', 'era', 'data_type', 'target_charles', 'target_elizabeth',
        'target_jordan', 'target_ken', 'target_frank', 'target_hillary'
    ],
                              axis=1)

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(train_bernie) if "feature" in f]
    X = train_bernie[features]
    Y = train_bernie['target_bernie']
    x_prediction = validation[features]
    ids = tournament['id']

    # This is your model that will learn to predict this target.
    model = linear_model.LogisticRegression(n_jobs=-1)
    print("# Training...")
    # Your model is trained on train_bernie
    model.fit(X, Y)

    print("# Predicting...")
    # Based on the model we can predict the probability of each row being
    # a bernie_target in the validation data.
    # The model returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    y_prediction = model.predict_proba(x_prediction)
    probabilities = y_prediction[:, 1]
    print("- probabilities:", probabilities[1:6])

    # We can see the probability does seem to be good at predicting the
    # true target correctly.
    print("- target:", validation['target_bernie'][1:6])
    print("- rounded probability:", [round(p) for p in probabilities][1:6])

    # But overall the accuracy is very low.
    correct = [
        round(x) == y
        for (x, y) in zip(probabilities, validation['target_bernie'])
    ]
    print("- accuracy: ", sum(correct) / float(validation.shape[0]))

    # The targets for each of the tournaments are very correlated.
    tournament_corr = np.corrcoef(validation['target_bernie'],
                                  validation['target_elizabeth'])
    print("- bernie vs elizabeth corr:", tournament_corr)
    # You can see that target_elizabeth is accurate using the bernie model as well.
    correct = [
        round(x) == y
        for (x, y) in zip(probabilities, validation['target_elizabeth'])
    ]
    print("- elizabeth using bernie:",
          sum(correct) / float(validation.shape[0]))

    # Numerai measures models on logloss instead of accuracy. The lower the logloss the better.
    # Numerai only pays models with logloss < 0.693 on the live portion of the tournament data.
    # Our validation logloss isn't very good.
    print("- validation logloss:",
          metrics.log_loss(validation['target_bernie'], probabilities))

    # To submit predictions from your model to Numerai, predict on the entire tournament data.
    x_prediction = tournament[features]
    y_prediction = model.predict_proba(x_prediction)
    results = y_prediction[:, 1]

    print("# Creating submission...")
    # Create your submission
    results_df = pd.DataFrame(data={'probability': results})
    joined = pd.DataFrame(ids).join(results_df)
    print("- joined:", joined.head())

    print("# Writing predictions to bernie_submissions.csv...")
    # Save the predictions out to a CSV file.
    joined.to_csv("bernie_submission.csv", index=False)
    # Now you can upload these predictions on https://numer.ai


"""
TIPS TO IMPROVE YOUR MODEL

1. Use eras
In this example, we dropped era column but you can use the era column to improve peformance across eras
You can take a model like the above and use it to generate probabilities on the training data, and
look at the the eras where your model was <0.693 and then build a new model on those bad eras to
combine with your main model. In this way, you may be hedged to the risk of bad eras in the future.
Advanced tip: To take this further, you could add the objective of doing consistenty well across all eras
directly into the objective function of your machine learning model.

2. Use feature importance
As per above, you don't want your model to rely too much on any particular type of era. Similarly, you
don't want your model to rely too much on any particular type of feature. If your model relies heavily on
one feature (in linear regression, some feature has very high coefficient), then if that feature doesn't work
in a particular era then your model will perform poorly. If your model balances its use of features then it is
more likely to be consistent across eras.

3. Use all the targets
As we saw above, a model trained on one target like target_bernie might be good at predicting another target
like target_elizabeth. Blending models built on each target could also improve your logloss and consistency.
"""

if __name__ == '__main__':
    main()
