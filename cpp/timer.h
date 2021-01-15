#pragma once

#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer(const std::string& message, const bool log_start = true) : message_(message) {
    start(log_start);
  }

  void start(const bool log_start) {
    if (log_start) {
      std::cout << message_ << "..." << std::endl;
    }
    start_ = std::chrono::steady_clock::now();
  }

  void end() {
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = end - start_;
    std::cout << message_ << " took " << std::chrono::duration<double, std::milli>(elapsed).count()
              << " ms." << std::endl;
  }

 private:
  std::chrono::time_point<std::chrono::steady_clock> start_;
  std::string message_;
};
