// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"


namespace lincs {

template<typename LinearProgram>
class OptimizeWeightsUsingLinearProgram : public LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy {
 public:
  OptimizeWeightsUsingLinearProgram(const PreprocessedLearningSet& preprocessed_learning_set_, ModelsBeingLearned& models_being_learned_) :
    LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy(true),
    preprocessed_learning_set(preprocessed_learning_set_),
    models_being_learned(models_being_learned_)
  {}

 public:
  void optimize_weights(unsigned model_indexes_begin, unsigned model_indexes_end) override;

 private:
  void optimize_model_weights(unsigned model_index);

 private:
  const PreprocessedLearningSet& preprocessed_learning_set;
  ModelsBeingLearned& models_being_learned;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__OPTIMIZE_WEIGHTS__LINEAR_PROGRAM_HPP
