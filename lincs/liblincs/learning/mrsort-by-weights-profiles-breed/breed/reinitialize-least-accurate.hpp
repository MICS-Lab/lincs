// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class ReinitializeLeastAccurate : public LearnMrsortByWeightsProfilesBreed::BreedingStrategy {
 public:
  explicit ReinitializeLeastAccurate(
    ModelsBeingLearned& models_being_learned_,
    LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy& profiles_initialization_strategy_,
    unsigned count_
  ) :
    LearnMrsortByWeightsProfilesBreed::BreedingStrategy(true),
    models_being_learned(models_being_learned_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    count(count_)
  {}

 public:
  void breed() override;

 private:
  ModelsBeingLearned& models_being_learned;
  LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy& profiles_initialization_strategy;
  unsigned count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
