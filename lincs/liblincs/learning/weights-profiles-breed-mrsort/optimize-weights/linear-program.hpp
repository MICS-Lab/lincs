// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

template<typename LinearProgram>
class OptimizeWeightsUsingLinearProgram : public WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
 public:
  OptimizeWeightsUsingLinearProgram(LearningData& learning_data_) : learning_data(learning_data_) {}

 public:
  void optimize_weights() override;

 private:
  void optimize_model_weights(unsigned model_index);

 private:
  LearningData& learning_data;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
