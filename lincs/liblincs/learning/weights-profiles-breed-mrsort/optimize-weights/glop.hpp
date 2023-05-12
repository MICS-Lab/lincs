#ifndef LINCS_LEARNING_OPTIMIZE_WEIGHTS_GLOP_HPP
#define LINCS_LEARNING_OPTIMIZE_WEIGHTS_GLOP_HPP

#include <memory>

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class OptimizeWeightsUsingGlop : public WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
 public:
  OptimizeWeightsUsingGlop(Models& models_) : models(models_) {}

 public:
  void optimize_weights() override;

 private:
  void optimize_model_weights(unsigned model_index);

  struct LinearProgram;

  std::shared_ptr<LinearProgram> make_internal_linear_program(const float epsilon, unsigned model_index);

  auto solve_linear_program(std::shared_ptr<LinearProgram> lp);

 private:
  Models& models;
};

}  // namespace lincs

#endif  // LINCS_LEARNING_OPTIMIZE_WEIGHTS_GLOP_HPP
