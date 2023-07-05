// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAtAccuracy : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(LearningData& learning_data_, unsigned target_accuracy_) : learning_data(learning_data_), target_accuracy(target_accuracy_) {}

 public:
  bool terminate() override;

 private:
  const LearningData& learning_data;
  const unsigned target_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP
