# Numerai C++ Pytorch Example

## Setup
The only requirement is [Pytorch](https://pytorch.org/) which can be met two ways:
1. Python dependency: `pip install torch torchvision`
2. LibTorch distribution from the link selector [here](https://pytorch.org/).

## Run.
```
./build.sh && ./build/numerai
```

## Optionally add build flags for clangd.
```
cp build/compile_commands.json .
```
