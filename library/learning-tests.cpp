// Copyright 2021 Vincent Jacques

#include <chrones.hpp>

#include "assign.hpp"
#include "generate.hpp"
#include "learning.hpp"
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

  auto result = Learning(
      learning_set,
      std::make_shared<TerminateAfterIterations>(3))
    .force_using_gpu()
    .set_random_seed(42)
    .set_models_count(15)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 89);

  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, std::vector<io::Model>(1, result.best_model));
  EXPECT_EQ(get_accuracy(models, 0), result.best_model_accuracy);
}

TEST(Learn, OnCpu) {
  std::mt19937 gen1(42);
  auto reference_model = generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = generate::learning_set(&gen2, reference_model, 100);

  auto result = Learning(
      learning_set,
      std::make_shared<TerminateAfterIterations>(2))
    .forbid_using_gpu()
    .set_random_seed(42)
    .set_models_count(5)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 78);

  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, std::vector<io::Model>(1, result.best_model));
  EXPECT_EQ(get_accuracy(models, 0), result.best_model_accuracy);
}

}  // namespace ppl
