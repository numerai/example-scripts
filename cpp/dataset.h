#pragma once

#include <torch/torch.h>

#include <iostream>
#include <vector>

class NumeraiDataset : public torch::data::datasets::Dataset<NumeraiDataset> {
 public:
  explicit NumeraiDataset(std::vector<torch::Tensor> features, std::vector<torch::Tensor> targets)
      : features_(std::move(features)), targets_(std::move(targets)) {}

  torch::data::Example<> get(size_t index) override {
    torch::Tensor sample_features = features_.at(index);
    torch::Tensor sample_target = targets_.at(index);
    return {sample_features.clone(), sample_target.clone()};
  };

  torch::optional<size_t> size() const override { return targets_.size(); };

 private:
  std::vector<torch::Tensor> features_;
  std::vector<torch::Tensor> targets_;
};
