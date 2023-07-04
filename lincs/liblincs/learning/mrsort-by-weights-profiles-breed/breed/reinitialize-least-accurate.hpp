// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class ReinitializeLeastAccurate : public LearnMrsortByWeightsProfilesBreed::BreedingStrategy {
 public:
  explicit ReinitializeLeastAccurate(
    LearningData& learning_data_,
    LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy& profiles_initialization_strategy_,
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
  LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy& profiles_initialization_strategy;
  unsigned count;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__BREED__REINITIALIZE_LEAST_ACCURATE_HPP
