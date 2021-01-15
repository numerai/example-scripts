#pragma once

#include <torch/torch.h>

struct Model : torch::nn::Module {
  Model() {
    fc1 = register_module("fc1", torch::nn::Linear(310, 100));
    fc2 = register_module("fc2", torch::nn::Linear(100, 1));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = fc1->forward(x);
    x = torch::relu(x);
    x = fc2->forward(x);
    return x;
  }

  torch::nn::Linear fc1 = nullptr;
  torch::nn::Linear fc2 = nullptr;
};
