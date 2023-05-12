#ifndef LINCS_LEARNING_TERMINATE_AT_ACCURACY_HPP
#define LINCS_LEARNING_TERMINATE_AT_ACCURACY_HPP

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

#endif  // LINCS_LEARNING_TERMINATE_AT_ACCURACY_HPP
