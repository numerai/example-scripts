#pragma once

#include <string>
#include <unordered_set>
#include <vector>

// Batch size.
constexpr int kBatchSize = 64;

// Number of training epochs.
constexpr int kEpoch = 1;

// Number of epochs to show losses after.
constexpr int kEpochInterval = 1;

// Learning rate.
constexpr float kLearningRate = 1e-3;

// Data type column name.
const std::string kDataType = "data_type";

// Path to save predictions to.
const std::string kPredictions = "predictions.csv";

// Path to training data.
const std::string kTrainingData = "../numerai_training_data.csv";

// Path to tournament data.
const std::string kTournamentData = "../numerai_tournament_data.csv";

// Number of rows per validation era.
const std::vector<int> kRowsPerValidationEra = {
    4573, 4658, 4609, 4630, 4698, 4682, 4688, 4636, 4705, 4756, 4814, 4812, 4970, 4981,
    5006, 4929, 5083, 5090, 5119, 5152, 5161, 5143, 4991, 5114, 5164, 5227, 5197, 5191};

// Tournament data types.
const std::unordered_set<std::string> kTournamentDataTypes = {"validation", "test", "live"};

// Training data types.
const std::unordered_set<std::string> kTrainingDataTypes = {"train"};

// Validation data types.
const std::unordered_set<std::string> kValidationDataTypes = {"validation"};
