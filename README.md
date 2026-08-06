# Numerai Example Scripts
The official place to start playing the Numerai tournaments.

Need help? [Find us on Discord.](https://discord.gg/numerai)

## Using Agents
Numerai is quickly developing open-source agent skills for you to use in the tournament. You can start architecting your very own AI scientist. For example:

```
git clone git@github.com:numerai/example-scripts
cd example-scripts && curl -sL http://numer.ai/install-mcp.sh | bash
codex exec --yolo "find the best neural network architecture to predict target_ender_60"
```

The maintained v5.3 examples pin `target_ender_60` explicitly. After the
in-place default-target cutover, generic `target` aliases Ender-60, while
`target_ender_20` remains available for intentionally reproducing the older
horizon. Explicit target names and the target-versioned cached model filenames
prevent an old Ender-20 model from being mistaken for the current example.

## Notebooks
We highly recommend getting started with Agents using the above section. But, if you're looking to kill some time on artisan data science, you can check out our tutorial notebooks here as well: 

- Hello Numerai: Start here if you are new! Explore the dataset and build your first model. <a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/numerai/hello_numerai.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Feature Neutralization: Learn how to measure feature risk and control it with feature neutralization. <a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/numerai/feature_neutralization.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Target Ensemble: Learn how to create an ensemble trained on different targets. <a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/numerai/target_ensemble.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

- Model Upload: A barebones example of how to build and upload your model to Numerai. <a target="_blank" href="https://colab.research.google.com/github/numerai/example-scripts/blob/master/numerai/example_model.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
