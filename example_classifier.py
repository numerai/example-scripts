#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing, model_selection, ensemble


def main():
    # Set seed for reproducibility
    np.random.seed(0)

    print("Loading data...")
    training_data = pd.read_csv('numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv('numerai_tournament_data.csv', header=0)

    Y = training_data['target']
    X = training_data.drop('target', axis=1)

    t_id = prediction_data['t_id']
    x_prediction = prediction_data.drop('t_id', axis=1)

    rf_classifier = ensemble.RandomForestClassifier(n_estimators=50, n_jobs=-1)

    print("Testing...")
    scores = model_selection.cross_val_score(rf_classifier, X, Y, cv=3, scoring="neg_log_loss")
    print("Mean cross validation log loss: {}".format(-1 * scores.mean()))

    print("Training...")
    rf_classifier.fit(X, Y)
    y_prediction = rf_classifier.predict_proba(x_prediction)

    print("Predicting...")
    # The random forest classifier returns two columns: [probability of 0, probability of 1]
    # We are just interested in the probability that the target is 1.
    results = y_prediction[:, 1]
    results_df = pd.DataFrame(data={'probability':results})
    joined = pd.DataFrame(t_id).join(results_df)

    print("Writing predictions to submission.csv")
    joined.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()
