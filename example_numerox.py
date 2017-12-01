#!/usr/bin/env python

"""
Example of how to prepare a submission for the Numerai tournament.
It uses Numerox which you can install with: pip install numerox
For more information see: https://github.com/kwgoodman/numerox
"""

import numerox as nx


def main():

    # download dataset from numerai
    nx.download_dataset('numerai_dataset.zip', verbose=True)

    # load numerai dataset
    data = nx.load_zip('numerai_dataset.zip', verbose=True)

    # we will use logistic regression; you will want to write your own model
    model = nx.model.logistic()

    # fit model with train data and make predictions for tournament data
    prediction = nx.production(model, data)

    # save predictions to csv file for later upload to numerai
    prediction.to_csv('logistic.csv', verbose=True)


if __name__ == '__main__':
    main()
