// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_WITHOUT_PROGRESS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_WITHOUT_PROGRESS_HPP

#include <chrono>

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterSecondsWithoutProgress : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterSecondsWithoutProgress(
    const ModelsBeingLearned& models_being_learned_,
    const float max_seconds_
  ) :
    models_being_learned(models_being_learned_),
    max_seconds(max_seconds_),
    last_progress_at(std::chrono::steady_clock::now()),
    previous_best_accuracy(0)
  {}

 public:
  bool terminate() override {
    const unsigned new_best_accuracy = models_being_learned.get_best_accuracy();
    if (new_best_accuracy > previous_best_accuracy) {
      last_progress_at = std::chrono::steady_clock::now();
      previous_best_accuracy = new_best_accuracy;
      return false;
    } else if (std::chrono::duration<float>(std::chrono::steady_clock::now() - last_progress_at).count() > max_seconds) {
      return true;
    } else {
      return false;
    }
  }

 private:
  const ModelsBeingLearned& models_being_learned;
  const float max_seconds;
  std::chrono::steady_clock::time_point last_progress_at;
  unsigned previous_best_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_SECONDS_WITHOUT_PROGRESS_HPP
