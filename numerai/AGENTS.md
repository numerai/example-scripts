# Numerai Tournament

This folder contains examples for the Numerai Tournament.

## Directory guide

- `agents/`: research framework and training pipeline (`python -m agents.code.modeling`)
- `v*/`: downloaded dataset files, usually ignored by Git
- `*.ipynb`: tutorials and model-upload examples

## Benchmark coverage

Historical benchmark predictions may cover fewer dates than the training or validation data. When joining benchmark predictions to targets, use the stock and date keys present in both files. Check coverage before scoring or applying benchmark-based transforms.
