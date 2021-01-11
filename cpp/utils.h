#pragma once

#include <torch/torch.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "constants.h"
#include "csv.h"
#include "dataset.h"
#include "model.h"
#include "timer.h"

std::vector<std::string> get_eras(const std::string& filename,
                                  const std::unordered_set<std::string>& data_types) {
  std::vector<std::string> eras;
  csv::CSVReader reader(filename);
  for (auto& row : reader) {
    // Filter by data type.
    if (data_types.find(row[kDataType].get<>()) == data_types.end()) {
      continue;
    }

    eras.push_back(row[kEra].get<>());
  }

  return eras;
}

std::vector<torch::Tensor> get_features(const std::string& filename,
                                        const std::unordered_set<std::string>& data_types) {
  std::vector<torch::Tensor> features;
  csv::CSVReader reader(filename);

  // Get feature names.
  const auto column_names = reader.get_col_names();
  std::vector<std::string> feature_names;
  feature_names.reserve(column_names.size());
  for (const auto& column_name : column_names) {
    if (column_name.find("feature") != std::string::npos) {
      feature_names.push_back(column_name);
    }
  }

  // Get values.
  std::vector<float> values;
  values.resize(feature_names.size());
  for (auto& row : reader) {
    // Filter by data type.
    if (data_types.find(row[kDataType].get<>()) == data_types.end()) {
      continue;
    }

    values.clear();
    for (const auto& feature_name : feature_names) {
      values.emplace_back(row[feature_name].get<float>());
    }

    // Convert to tensor.
    torch::Tensor tensor =
        torch::from_blob(values.data(), {1, static_cast<long>(values.size())}).clone();
    features.push_back(tensor);
  }

  return features;
}

std::vector<torch::Tensor> get_targets(const std::string& filename,
                                       const std::unordered_set<std::string>& data_types) {
  std::vector<torch::Tensor> targets;
  csv::CSVReader reader(filename);
  for (auto& row : reader) {
    // Filter by data type.
    if (data_types.find(row[kDataType].get<>()) == data_types.end()) {
      continue;
    }

    targets.push_back(torch::tensor({row["target"].get<float>()}));
  }

  return targets;
}

NumeraiDataset get_training_data() {
  const auto features = get_features(kTrainingData, kTrainingDataTypes);
  const auto targets = get_targets(kTrainingData, kTrainingDataTypes);

  NumeraiDataset numerai_dataset(features, targets);
  return numerai_dataset;
}

// Calculates Pearson correlation coefficient based on single pass formula from:
// https://en.wikipedia.org/wiki/Pearson_correlation_coefficient#For_a_sample
double pearson(const std::vector<float>& x, const std::vector<float>& y) {
  assert(x.size() == y.size());
  const int n = x.size();

  long double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0;
  long double squared_sum_x = 0.0, squared_sum_y = 0.0;

  for (int i = 0; i < n; ++i) {
    sum_x += x[i];
    sum_y += y[i];
    sum_xy += (x[i] * y[i]);
    squared_sum_x += (x[i] * x[i]);
    squared_sum_y += (y[i] * y[i]);
  }

  const long double numerator = (n * sum_xy) - (sum_x * sum_y);
  const long double denominator =
      std::sqrt((n * squared_sum_x - (sum_x * sum_x)) * (n * squared_sum_y - (sum_y * sum_y)));
  return numerator / denominator;
}

// Ranks elements starting from 0.
std::vector<int> rank(std::vector<float> input) {
  std::vector<float> sorted = input;
  std::stable_sort(sorted.begin(), sorted.end());

  std::unordered_map<float, int> value_to_position;
  std::unordered_map<float, int> value_to_count;
  for (int i = 0; i < sorted.size(); ++i) {
    ++value_to_count[sorted[i]];
    value_to_position[sorted[i]] = i;
  }

  std::vector<int> ranks;
  ranks.reserve(input.size());
  for (int i = 0; i < input.size(); ++i) {
    ranks.push_back(value_to_position[input[i]] - (value_to_count[input[i]] - 1));
    --value_to_count[input[i]];
  }

  return ranks;
}

std::vector<float> rescale(const std::vector<float>& input) {
  const auto minmax = std::minmax_element(input.begin(), input.end());
  const float min = *minmax.first;
  const float max = *minmax.second;

  std::vector<float> output;
  output.reserve(input.size());
  for (const auto& e : input) {
    output.push_back((e - min) / (max - min));
  }

  return output;
}

void save_predictions(const std::vector<float>& predictions_unscaled) {
  // Rescale to [0, 1].
  const auto predictions = rescale(predictions_unscaled);

  std::ofstream file(kPredictions);
  auto writer = csv::make_csv_writer(file);

  // Write header.
  writer << std::vector<std::string>({"id", "prediction"});

  // Write predictions.
  csv::CSVReader reader(kTournamentData);
  int i = 0;
  for (const auto& row : reader) {
    writer << std::vector<std::string>({row["id"].get<>(), std::to_string(predictions[i++])});
  }
}

double score(const std::vector<float>& x, const std::vector<float>& y) {
  const auto rank_x_int = rank(x);
  const std::vector<float> rank_x(rank_x_int.begin(), rank_x_int.end());
  return pearson(rank_x, y);
}

void show_metrics(const std::vector<float>& predictions, const std::vector<float> targets,
                  const std::vector<std::string>& eras) {
  // Get correlations per era.
  std::vector<double> correlations;
  std::vector<float> predictions_per_era;
  std::vector<float> targets_per_era;

  for (int i = 0; i <= eras.size(); ++i) {
    // Process the just ended era.
    if (i == eras.size() || (i > 0 && eras[i] != eras[i - 1])) {
      const auto correlation = score(predictions_per_era, targets_per_era);
      correlations.push_back(correlation);

      predictions_per_era.clear();
      targets_per_era.clear();
    }

    if (i < eras.size()) {
      predictions_per_era.push_back(predictions[i]);
      targets_per_era.push_back(targets[i]);
    }
  }

  // Compute metrics.
  const double validation_correlation_mean =
      std::accumulate(correlations.begin(), correlations.end(), 0.0) / correlations.size();
  const double validation_correlation_std_dev = std::sqrt(std::accumulate(
      correlations.begin(), correlations.end(), 0.0,
      [&validation_correlation_mean, &correlations](double accumulator, const double& e) {
        return accumulator + ((e - validation_correlation_mean) *
                              (e - validation_correlation_mean) / correlations.size());
      }));
  const double validation_correlation_sharpe =
      validation_correlation_mean / validation_correlation_std_dev;

  // Show metrics.
  std::cout << "validation_correlation_mean = " << validation_correlation_mean << std::endl;
  std::cout << "validation_correlation_sharpe = " << validation_correlation_sharpe << std::endl;
}

void train(std::shared_ptr<Model> net, NumeraiDataset& numerai_dataset) {
  // Make data loader.
  auto dataset = numerai_dataset.map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
      std::move(dataset), kBatchSize);

  // Set up optimizer.
  const torch::optim::SGDOptions options(kLearningRate);
  torch::optim::SGD optimizer(net->parameters(), options);

  // Train.
  for (int epoch = 0; epoch < kEpoch; ++epoch) {
    for (auto& batch : *data_loader) {
      auto x = batch.data.squeeze();
      auto y = batch.target.squeeze();

      // Zero gradients.
      optimizer.zero_grad();

      // Predict.
      auto predictions = net->forward(x);

      // Compute loss.
      const auto loss = torch::mse_loss(predictions, y);

      // Show loss.
      if (epoch % kEpochInterval == 0) {
        std::cout << "Epoch: " << epoch << " loss: " << loss.item<double>() << std::endl;
      }

      // Compute gradients and update weights.
      loss.backward();
      optimizer.step();
    }
  }
}

void test(std::shared_ptr<Model> net) {
  Timer timer("Loading validation data");
  const auto eras = get_eras(kTournamentData, kValidationDataTypes);
  const auto features = get_features(kTournamentData, kValidationDataTypes);
  const auto targets = get_targets(kTournamentData, kValidationDataTypes);
  timer.end();

  // Concatenate.
  const auto x = torch::cat(features);
  const auto y = torch::cat(targets);

  // Predict.
  timer = Timer("Predicting validation");
  auto predictions = net->forward(x);
  timer.end();

  const std::vector<float> predictions_vector(predictions.data_ptr<float>(),
                                              predictions.data_ptr<float>() + predictions.numel());
  const std::vector<float> y_vector(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());

  // Show metrics.
  timer = Timer("Calculating correlation");
  show_metrics(predictions_vector, y_vector, eras);
  timer.end();
}
