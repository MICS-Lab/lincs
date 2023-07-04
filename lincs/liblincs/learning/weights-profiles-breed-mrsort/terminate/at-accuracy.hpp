// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class TerminateAtAccuracy : public WeightsProfilesBreedMrSortLearning::TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(LearningData& learning_data_, unsigned target_accuracy_) : learning_data(learning_data_), target_accuracy(target_accuracy_) {}

 public:
  bool terminate() override;

 private:
  LearningData& learning_data;
  unsigned target_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP
