```
      ___           ___           ___           ___           ___           ___                 
     /\__\         /\__\         /\__\         /\  \         /\  \         /\  \          ___   
    /::|  |       /:/  /        /::|  |       /::\  \       /::\  \       /::\  \        /\  \  
   /:|:|  |      /:/  /        /:|:|  |      /:/\:\  \     /:/\:\  \     /:/\:\  \       \:\  \ 
  /:/|:|  |__   /:/  /  ___   /:/|:|__|__   /::\~\:\  \   /::\~\:\  \   /::\~\:\  \      /::\__\
 /:/ |:| /\__\ /:/__/  /\__\ /:/ |::::\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\  __/:/\/__/
 \/__|:|/:/  / \:\  \ /:/  / \/__/~~/:/  / \:\~\:\ \/__/ \/_|::\/:/  / \/__\:\/:/  / /\/:/  /   
     |:/:/  /   \:\  /:/  /        /:/  /   \:\ \:\__\      |:|::/  /       \::/  /  \::/__/    
     |::/  /     \:\/:/  /        /:/  /     \:\ \/__/      |:|\/__/        /:/  /    \:\__\    
     /:/  /       \::/  /        /:/  /       \:\__\        |:|  |         /:/  /      \/__/    
     \/__/         \/__/         \/__/         \/__/         \|__|         \/__/                
```

These are the example machine learning scripts included with the download of [Numerai's data](https://numer.ai/).

# Contents
* [Quick Start](#quick-start)
* [Datasets](#data-sets)
* [Next Steps](#next-steps)
  * [Automating Submissions](#automating-submissions)
  * [Feature Engineering](#feature-engineering)
  * [Using Multiple Targets](#using-multiple-targets)
* [FAQ](#faq)  
* [Help and Discussion](#help-and-discussion)

# Quick Start
```
pip install -r requirements.txt
python example_model.py
```

The example script model will produce a `validation_predictions.csv` file which you can upload at 
https://numer.ai/tournament to get model diagnostics.

> TIP: The example_model.py script takes ~45-60 minutes to run. In the mean time, you can upload
> `example_diagnostic_predictions.csv` to get diagnostics immediately. We also recommend 
> reading the [Next Steps](#next-steps) section

![upload-diagnostics](https://github.com/numerai/example-scripts/blob/chris/update-example-scripts/media/upload_diagnostics.gif)

If the current round is open (Saturday 18:00 UTC through Monday 14:30 UTC), you can submit predictions and start
getting results on live tournament data. You can create your submission by uploading the `tournament_predictions.csv`
file at https://numer.ai/tournament

![upload-tournament](https://github.com/numerai/example-scripts/blob/chris/update-example-scripts/media/upload_tournament.gif)

# Datasets

### numerai_training_data

- Description: Data to train your model.

- Contents: 

    - "id": string labels of obfuscated stock IDs

    - "era": string labels of points in time for a block of IDs

    - "data_type": string label "train"

    - "feature_...": floating-point numbers,
      obfuscated characteristics for each stock ID

    - "target": floating-point numbers,
      the relative performance of that stock during that era

- Notes: Check out the [examples repo](https://github.com/numerai/example-scripts) to learn how to train a model and the 
  analysis_and_tips notebook in numerai_datasets.zip to learn how to explore this data.


### numerai_validation_data

- Description: Data that your model uses to generate predictions for diagnostics.

- Contents: 

    - "id": string labels of obfuscated stock IDs

    - "era": string labels of points in time for a block of IDs

    - "data_type": string label "validation"

    - "feature_...": floating-point numbers,
      obfuscated characteristics for each stock ID

    - "target": floating-point numbers,
      the relative performance of that stock during that era

- Notes: Diagnostics are useful to help determine the efficacy of your model. These
  are not predictive of performance, but are helpful when iterating on a model. 


### numerai_tournament_data

- Description: Data that your model uses to generate predictions to submit each round.

- Contents: 

    - "id": string labels of obfuscated stock IDs

    - "era": string labels of points in time for a block of IDs

    - "data_type": string labels "test" and "live"

    - "feature_...": floating-point numbers,
      obfuscated characteristics for each stock ID

    - "target": NaN (not-a-number),
      should be filled in with floating-point numbers by your model

- Notes: This file changes every week, so make sure you're predicting on the most
  recent version of this file each round.

### example_predictions

- Description: An example of predictions on the numerai_tournament_data file.

- Contents:

    - "id": string labels of obfuscated stock IDs

    - "prediction": floating-point numbers between 0 and 1 (exclusive)

- Notes: Useful for ensuring you can make a submission and debugging your prediction 
  file if you receive an error from the submissions API. This is what your submissions
  should look like (same ids and data types).


### example_validation_predictions

- Description: An example of predictions on the numerai_validation_data file.

- Contents:

    - "id": string labels of obfuscated stock IDs

    - "prediction": floating-point numbers between 0 and 1 (exclusive)

- Notes: Useful for ensuring you can get diagnostics and debugging your prediction 
  file if you receive an error from the diagnostics API. This is what your uploads
  should look like when using the diagnostics section under a model's "More" dropdown 
  in the [models page](numer.ai/models).

# Next Steps
The supplied `example_model.py` script is the baseline for a model that can provide useful predictions to our 
metamodel. We have outlined some next steps which you can use to improve your 
model performance and payouts.



## Automating Submissions
The first step in automating submissions is to create API keys for your model (https://numer.ai/account) and provide them when creating the
NumerAPI client:

```python
example_public_id = "somepublicid"
example_secret_key = "somesecretkey"
napi = numerapi.NumerAPI(example_public_id, example_secret_key)
```

After instantiating the NumerAPI client with API keys, you can then upload your submissions programmatically:

```python
# upload predictions
model_id = napi.get_models()['your_model_name']
napi.upload_predictions("tournament_predictions.csv", model_id=model_id)
```


The recommended setup for a fully automated submission process is to use Numerai Compute. Please see the 
[Numerai CLI documentation](https://github.com/numerai/numerai-cli) for instructions on how to deploy your 
models to AWS

## Feature engineering
> TODO: show code snippets from example_script where we find riskiest features and do feature neutralization

> TODO: ideas/hints on what other types of feature engineering could be done

For more information on neutralization, see this forum post by user JRB https://forum.numer.ai/t/model-diagnostics-feature-exposure/899

## Using multiple targets
> TODO: show code snippets from example_script where we predict on multiple models and then ensemble. 
> Also show the validation diagnostics for each

> TODO: explain what each target is!
 
> TODO: ideas/hints on how to combine multiple models

# FAQ
### Are my diagnostic metrics good?
Diagnostics are an indicator of how well your model will perform, but previous performance is not necessarily 
an indicator of future success. In the long run, it is only possible to know what your model performance is by 
submitting live predictions.
### How much can I make by staking? 
Your payouts are determined by your model's performance, however you can get a general sense of how much our 
users are making by looking at the aggregate statistics on the tournament leaderboard: https://numer.ai/tournament
### How to get NMR
You can buy NMR with a debit card or bank transfer on [Coinbase](https://coinbase.com) and 
[Binance](https://www.binance.com/) in supported regions.

Alternatively, acquire Bitcoin (or another cryptocurrency) and convert it to NMR on an exchange like 
Coinbase or Binance (or a decentralized exchange like Uniswap).

You can then send your NMR to your Numerai deposit address, which you will find on your Numerai wallet page.

### How do I stake?
Once you have NMR in your wallet, you can start staking by clicking on the Manage Stake button on the website. 
This will open a page that will allow you to increase your stake as well as choosing which of your performance 
scores you would like to stake on.

### Can I do time series modeling? 

# Help and Discussion
For help and discussion, join our community chat (http://community.numer.ai/) or our forums (https://forum.numer.ai/)