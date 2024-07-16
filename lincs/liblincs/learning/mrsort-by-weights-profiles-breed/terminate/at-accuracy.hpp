// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateAtAccuracy : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(ModelsBeingLearned& models_being_learned_, unsigned target_accuracy_) : models_being_learned(models_being_learned_), target_accuracy(target_accuracy_) {}

 public:
  bool terminate() override;

 private:
  const ModelsBeingLearned& models_being_learned;
  const unsigned target_accuracy;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__AT_ACCURACY_HPP
