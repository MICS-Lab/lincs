// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterIterations : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterIterations(const LearningData& learning_data_, const unsigned max_iterations_count_) : learning_data(learning_data_), max_iterations_count(max_iterations_count_) {}

 public:
  bool terminate() override {
    return learning_data.iteration_index >= max_iterations_count - 1;
  }

 private:
  const LearningData& learning_data;
  const unsigned max_iterations_count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
