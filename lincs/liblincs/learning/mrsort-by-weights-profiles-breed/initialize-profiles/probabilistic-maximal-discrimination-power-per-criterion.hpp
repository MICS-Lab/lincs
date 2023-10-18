// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP

#include <utility>

#include "../../mrsort-by-weights-profiles-breed.hpp"
#include "../../../randomness-utils.hpp"

namespace lincs {

class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(LearningData& learning_data_);

 public:
  void initialize_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) override;

 private:
  std::map<unsigned, double> get_candidate_probabilities(
    unsigned criterion_index,
    unsigned profile_index
  );

 private:
  LearningData& learning_data;
  std::vector<std::vector<ProbabilityWeightedGenerator<unsigned>>> rank_generators;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
