// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class ReinitializeLeastAccurate : public WeightsProfilesBreedMrSortLearning::BreedingStrategy {
 public:
  explicit ReinitializeLeastAccurate(
    LearningData& learning_data_,
    WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy& profiles_initialization_strategy_,
    unsigned count_
  ) :
    learning_data(learning_data_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    count(count_)
  {}

 public:
  void breed() override;

 private:
  LearningData& learning_data;
  WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy& profiles_initialization_strategy;
  unsigned count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
