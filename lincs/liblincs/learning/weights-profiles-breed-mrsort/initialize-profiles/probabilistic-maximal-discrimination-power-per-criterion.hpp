// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP

#include "../../weights-profiles-breed-mrsort.hpp"
#include "../../../randomness-utils.hpp"

namespace lincs {

class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(Models& models_);

 public:
  void initialize_profiles(
    std::vector<unsigned>::const_iterator model_indexes_begin,
    const std::vector<unsigned>::const_iterator model_indexes_end
  ) override;

 private:
  std::map<float, double> get_candidate_probabilities(
    unsigned criterion_index,
    unsigned profile_index
  );

 private:
  Models& models;
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> generators;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
