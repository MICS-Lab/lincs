// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_HPP

#include <chrono>

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterSeconds : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterSeconds(const float max_seconds_) : max_seconds(max_seconds_), started_at(std::chrono::steady_clock::now()) {}

 public:
  bool terminate() override {
    return std::chrono::duration<float>(std::chrono::steady_clock::now() - started_at).count() > max_seconds;
  }

 private:
  const float max_seconds;
  const std::chrono::steady_clock::time_point started_at;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_HPP
