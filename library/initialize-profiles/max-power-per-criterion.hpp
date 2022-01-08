// Copyright 2021-2022 Vincent Jacques

#ifndef INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_
#define INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_

#include <memory>
#include <vector>

#include "../initialize-profiles.hpp"
#include "../randomness.hpp"


namespace ppl {

/*
Implement 3.3.2 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(RandomNumberGenerator, const Models<Host>&);

  void initialize_profiles(
    std::shared_ptr<Models<Host>>,
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end) override;

 private:
  RandomNumberGenerator _random;
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> _generators;
};

// class InitializeProfilesForDeterministicMaximalDiscriminationPowerPerCriterion :
// public ProfilesInitializationStrategy {
//  public:
//   void initialize_profiles(
//     Models<Host>* models,
//     uint iteration_index,
//     std::vector<uint>::const_iterator model_indexes_begin,
//     std::vector<uint>::const_iterator model_indexes_end) override;
// };

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_
