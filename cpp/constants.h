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

// Era column name.
const std::string kEra = "era";

// Path to save predictions to.
const std::string kPredictions = "predictions.csv";

// Path to training data.
const std::string kTrainingData = "../numerai_training_data.csv";

// Path to tournament data.
const std::string kTournamentData = "../numerai_tournament_data.csv";

// Tournament data types.
const std::unordered_set<std::string> kTournamentDataTypes = {"validation", "test", "live"};

// Training data types.
const std::unordered_set<std::string> kTrainingDataTypes = {"train"};

// Validation data types.
const std::unordered_set<std::string> kValidationDataTypes = {"validation"};
