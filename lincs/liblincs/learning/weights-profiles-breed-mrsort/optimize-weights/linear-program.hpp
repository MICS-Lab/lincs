// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

template<typename LinearProgram>
class OptimizeWeightsUsingLinearProgram : public WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
 public:
  OptimizeWeightsUsingLinearProgram(Models& models_) : models(models_) {}

 public:
  void optimize_weights() override;

 private:
  void optimize_model_weights(unsigned model_index);

 private:
  Models& models;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
