// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterIterations : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterIterations(const ModelsBeingLearned& models_being_learned_, const unsigned max_iterations_count_) : models_being_learned(models_being_learned_), max_iterations_count(max_iterations_count_) {}

 public:
  bool terminate() override {
    return models_being_learned.iteration_index >= max_iterations_count - 1;
  }

 private:
  const ModelsBeingLearned& models_being_learned;
  const unsigned max_iterations_count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
