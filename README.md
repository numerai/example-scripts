# Numerai Example Scripts

A collection of scripts and notebooks to help you get started quickly. 

Need help? Find us on Discord:

[![](https://dcbadge.vercel.app/api/server/numerai)](https://discord.gg/numerai)


## Notebooks 

These these notebooks run on Google Colab's free tier.

### Hello Numerai
<a target="_blank" href="https://colab.research.google.com/github/numerai/hello-numerai/blob/master/hello_numerai.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Start here if you are new! Explore the dataset and build your first model. 
 
### Analysis and Tips
<a target="_blank" href="https://colab.research.google.com/github/numerai/hello-numerai/blob/master/analysis_and_tips.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

Best practices and common mistakes to avoid when apply machine learning to Numerai's dataset.

### Model Upload
<a target="_blank" href="https://colab.research.google.com/github/numerai/hello-numerai/blob/master/model_upload.ipynb">
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

### Train latest example model training on v4.1 data 
```
python example_model_sunshine.py
```

### System Requirements
- Minimum: 16GB RAM and 4 core CPU
- Recommended: 32GB RAM and 8 core CPU