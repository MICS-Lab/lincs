#ifndef LINCS_LEARNING_INITIALIZE_PROFILES_MAX_DISC_PER_CRIT_HPP
#define LINCS_LEARNING_INITIALIZE_PROFILES_MAX_DISC_PER_CRIT_HPP

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

#endif  // LINCS_LEARNING_INITIALIZE_PROFILES_MAX_DISC_PER_CRIT_HPP
