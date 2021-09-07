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

These are the example machine learning scripts included with the download of [Numerai's data](https://numer.ai/learn).

# Contents
* [Quick Start](#quick-start)
* [Next Steps](#next-steps)
  * [Understanding the Data](#understanding-data)
  * [Automating Submissions](#automating-submissions)
  * [Advanced Models](#next-steps)
  * [Staking](#staking)
  * [Feature Engineering](#feature-engineering)
* [FAQ](#faq) 

# Quick Start
```
pip install -r requirements.txt
python example_model.py
```

The example script model will produce a `validation_predictions.csv` file which you can upload at https://numer.ai/tournament
to get model diagnostics.

> TIP: The example_model.py script takes approximately 45 minutes to run. In the mean time, we recommend reading the [Understanding the Data](#understanding-data) section

![upload-predictions](https://github.com/numerai/example-scripts/blob/chris/update-example-scripts/media/upload_predictions.gif)

If the current round is open (Saturday 18:00 UTC through Monday 14:30 UTC), you can submit predictions and start
getting results on live tournament data. You can create your submission by uploading the `test_and_live_predictions.csv`
file at https://numer.ai/tournament

### TODO: submit live predictions gif here

# Next Steps
Now that you have your first model diagnostics, we have outlined some next steps which you can take to improve your 
model performance.
## Understanding Data

* What are the different data files
* What are the eras
## Automating Submissions

* Show example model script where you put in API keys
* Numerai CLI

## Advanced models
## Feature engineering
## Staking
## Signals
# FAQ
### Are my diagnostic metrics good?
### What are eras? 
### How much can I make by staking? 
### How do I stake?
### Can I do time series modeling? 
