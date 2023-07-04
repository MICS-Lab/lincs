// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class ReinitializeLeastAccurate : public WeightsProfilesBreedMrSortLearning::BreedingStrategy {
 public:
  explicit ReinitializeLeastAccurate(
    Models& models_,
    WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy& profiles_initialization_strategy_,
    unsigned count_
  ) :
    models(models_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    count(count_)
  {}

 public:
  // @todo Move to .cpp file
  void breed() override {
    profiles_initialization_strategy.initialize_profiles(models.model_indexes.begin(), models.model_indexes.begin() + count);
  }

 private:
  Models& models;
  WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy& profiles_initialization_strategy;
  unsigned count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
