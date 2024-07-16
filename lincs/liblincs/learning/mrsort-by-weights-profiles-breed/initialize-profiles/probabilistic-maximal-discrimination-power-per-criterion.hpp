// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP

#include <utility>

#include "../../mrsort-by-weights-profiles-breed.hpp"
#include "../../../randomness-utils.hpp"

namespace lincs {

class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(const PreprocessedLearningSet& preprocessed_learning_set, ModelsBeingLearned& models_being_learned_);

 public:
  void initialize_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) override;

 private:
  std::map<unsigned, double> get_candidate_probabilities_for_low_ranks(
    unsigned criterion_index,
    unsigned boundary_index
  );

  std::map<unsigned, double> get_candidate_probabilities_for_high_ranks(
    unsigned criterion_index,
    unsigned boundary_index
  );

 private:
  const PreprocessedLearningSet& preprocessed_learning_set;
  ModelsBeingLearned& models_being_learned;
  std::vector<std::vector<ProbabilityWeightedGenerator<unsigned>>> low_rank_generators;
  std::vector<std::vector<ProbabilityWeightedGenerator<unsigned>>> high_rank_generators;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__INITIALIZE_PROFILES__PROBABILISTIC_MAXIMAL_DISCRIMINATION_POWER_PER_CRITERION_HPP
