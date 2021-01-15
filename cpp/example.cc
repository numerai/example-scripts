// Pytorch C++ example.

#include <torch/torch.h>

#include <memory>
#include <unordered_set>

#include "constants.h"
#include "dataset.h"
#include "model.h"
#include "timer.h"
#include "utils.h"

int main() {
  // Make net.
  auto net = std::make_shared<Model>();

  // Load training data.
  Timer timer("Loading training data");
  auto training_data = get_training_data();
  timer.end();

  // Train.
  timer = Timer("Training model");
  train(net, training_data);
  timer.end();

  // Test on validation.
  test(net);

  // Load tournament data.
  timer = Timer("Loading tournament data");
  const auto features = get_features(kTournamentData, kTournamentDataTypes);
  timer.end();

  // Predict tournament data.
  timer = Timer("Predicting tournament data");
  const auto x = torch::cat(features);
  const auto y = net->forward(x);
  timer.end();

  // Save predictions.
  timer = Timer("Saving predictions");
  const std::vector<float> y_vector(y.data_ptr<float>(), y.data_ptr<float>() + y.numel());
  save_predictions(y_vector);
  timer.end();
}
