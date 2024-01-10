// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__COMPOSITE_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__COMPOSITE_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

class TerminateWhenAny : public LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
 public:
  explicit TerminateWhenAny(const std::vector<LearnMrsortByWeightsProfilesBreed::TerminationStrategy*>& termination_strategies_) : termination_strategies(termination_strategies_) {}

 public:
  bool terminate() override;

 private:
  const std::vector<LearnMrsortByWeightsProfilesBreed::TerminationStrategy*> termination_strategies;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__TERMINATE__COMPOSITE_HPP
