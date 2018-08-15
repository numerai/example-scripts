#!/usr/bin/env python

import numerox as nx


def numerox_example():
    """
    Example of how to prepare a submission for the Numerai tournament.
    It uses Numerox which you can install with: pip install numerox
    For more information see: https://github.com/kwgoodman/numerox
    """

    # download dataset from numerai
    nx.download('numerai_dataset.zip', verbose=True)

    # load numerai dataset
    data = nx.load_zip('numerai_dataset.zip', verbose=True)

    # we will use logistic regression; you will want to write your own model
    model = nx.logistic()

    # fit model with train data and make predictions for tournament data
    prediction = nx.production(model, data, tournament='bernie')

    # save predictions to csv file
    prediction.to_csv('logistic.csv', tournament='bernie', verbose=True)

    # upload predictions to Numerai to enter the tournament
    #
    # you create the public_id and secret_key on the Numerai website
    # nx.upload('logistic.csv', tournament='bernie', public_id, secret_key)


if __name__ == '__main__':
    numerox_example()
