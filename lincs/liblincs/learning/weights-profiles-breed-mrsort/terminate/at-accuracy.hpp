// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class TerminateAtAccuracy : public WeightsProfilesBreedMrSortLearning::TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(unsigned target_accuracy) : _target_accuracy(target_accuracy) {}

 public:
  bool terminate(unsigned, unsigned) override;

 private:
  unsigned _target_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__TERMINATE__AT_ACCURACY_HPP
