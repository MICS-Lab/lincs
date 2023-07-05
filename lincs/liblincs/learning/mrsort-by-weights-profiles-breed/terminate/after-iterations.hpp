// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAfterIterations : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAfterIterations(const LearningData& learning_data_, const unsigned max_iteration_index_) : learning_data(learning_data_), max_iteration_index(max_iteration_index_) {}

 public:
  bool terminate() override {
    return learning_data.iteration_index > max_iteration_index;
  }

 private:
  const LearningData& learning_data;
  const unsigned max_iteration_index;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AFTER_ITERATIONS_HPP
