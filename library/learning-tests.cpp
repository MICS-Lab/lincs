// Copyright 2021-2022 Vincent Jacques

#include <chrones.hpp>

#include "assign.hpp"
#include "generate.hpp"
#include "improve-profiles/heuristic-for-accuracy.hpp"
#include "initialize-profiles/max-power-per-criterion.hpp"
#include "learning.hpp"
#include "optimize-weights/glop.hpp"
#include "terminate/iterations.hpp"
#include "test-utils.hpp"


CHRONABLE("learning-tests")

namespace ppl {

TEST(Learn, OnGpu) {
  std::mt19937 gen1(42);
  auto reference_model = generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = generate::learning_set(&gen2, reference_model, 100);

  auto host_domain = Domain<Host>::make(learning_set);
  auto host_models = Models<Host>::make(host_domain, 15);

  auto device_domain = host_domain.clone_to<Device>();
  auto device_models = host_models.clone_to<Device>(device_domain);

  auto result = Learning(
      host_domain, &host_models,
      {},
      std::make_shared<InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion>(host_models),
      std::make_shared<OptimizeWeightsUsingGlop>(),
      std::make_shared<ImproveProfilesWithAccuracyHeuristicOnGpu>(&host_models, &device_models),
      std::make_shared<TerminateAfterIterations>(3))
    .set_random_seed(42)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 89);

  auto best_models = Models<Host>::make(host_domain, std::vector<io::Model>(1, result.best_model));
  EXPECT_EQ(get_accuracy(best_models, 0), result.best_model_accuracy);
}

TEST(Learn, OnCpu) {
  std::mt19937 gen1(42);
  auto reference_model = generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = generate::learning_set(&gen2, reference_model, 100);

  auto host_domain = Domain<Host>::make(learning_set);
  auto host_models = Models<Host>::make(host_domain, 5);

  auto result = Learning(
      host_domain, &host_models,
      {},
      std::make_shared<InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion>(host_models),
      std::make_shared<OptimizeWeightsUsingGlop>(),
      std::make_shared<ImproveProfilesWithAccuracyHeuristicOnCpu>(&host_models),
      std::make_shared<TerminateAfterIterations>(2))
    .set_random_seed(42)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 78);

  auto best_models = Models<Host>::make(host_domain, std::vector<io::Model>(1, result.best_model));
  EXPECT_EQ(get_accuracy(best_models, 0), result.best_model_accuracy);
}

}  // namespace ppl
