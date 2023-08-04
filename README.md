# Numerai Example Scripts

A collection of scripts and notebooks to help you get started quickly. 

Need help? Find us on Discord:

[![](https://dcbadge.vercel.app/api/server/numerai)](https://discord.gg/numerai)


## Notebooks 

Try running these notebooks on Google Colab's free tier!

### Hello Numerai
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/hello_numerai.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Start here if you are new! Explore the dataset and build your first model. 

### Feature Neutralization
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/feature_neutralization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Explore the tradeoff between risk and performance. Learn how to measure risk with feature exposure and control it with feature neutralization.

### Target Ensemble
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/target_ensemble.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Learn how to create an ensemble trained on different targets.

### Model Upload
<a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/model_upload.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

A barebones example of how to upload your model to Numerai.


## Scripts
### Setup
```
pip install -U pip && pip install -r requirements.txt
```

### Train basic example model on v4 data
```
python example_model.py
```

### Train advanced example model on v4 data 
```
python example_model_advanced.py
```

### Train latest example model training on v4.1 data in int8 format 
```
python example_model_sunshine.py
```

### System Requirements
- Minimum: 16GB RAM and 4 core CPU
- Recommended: 32GB RAM and 8 core CPU