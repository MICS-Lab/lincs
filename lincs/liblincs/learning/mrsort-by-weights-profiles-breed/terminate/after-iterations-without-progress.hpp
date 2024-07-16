// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_WITHOUT_PROGRESS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_WITHOUT_PROGRESS_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterIterationsWithoutProgress : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterIterationsWithoutProgress(
    const ModelsBeingLearned& models_being_learned_,
    const unsigned max_iterations_count_
  ) :
    models_being_learned(models_being_learned_),
    max_iterations_count(max_iterations_count_),
    last_progress_iteration_index(0),
    previous_best_accuracy(0)
  {}

 public:
  bool terminate() override {
    const unsigned new_best_accuracy = models_being_learned.get_best_accuracy();
    if (new_best_accuracy > previous_best_accuracy) {
      last_progress_iteration_index = models_being_learned.iteration_index;
      previous_best_accuracy = new_best_accuracy;
      return false;
    } else if (models_being_learned.iteration_index - last_progress_iteration_index >= max_iterations_count) {
      return true;
    } else {
      return false;
    }
  }

 private:
  const ModelsBeingLearned& models_being_learned;
  const unsigned max_iterations_count;
  unsigned last_progress_iteration_index;
  unsigned previous_best_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_WITHOUT_PROGRESS_HPP
