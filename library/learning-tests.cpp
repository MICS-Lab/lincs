// Copyright 2021 Vincent Jacques

#include "assign.hpp"
#include "generate.hpp"
#include "learning.hpp"
#include "test-utils.hpp"


TEST(Learn, OnGpu) {
  std::mt19937 gen1(42);
  auto reference_model = ppl::generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = ppl::generate::learning_set(&gen2, reference_model, 100);

  auto result = ppl::learning::Learning(learning_set)
    .set_max_iterations(3)
    .set_target_accuracy(learning_set.alternatives_count)
    .force_using_gpu()
    .set_random_seed(42)
    .set_models_count(15)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 89);

  auto domain = ppl::Domain<Host>::make(learning_set);
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, result.best_model));
  EXPECT_EQ(ppl::get_accuracy(models, 0), result.best_model_accuracy);
}


TEST(Learn, OnCpu) {
  std::mt19937 gen1(42);
  auto reference_model = ppl::generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = ppl::generate::learning_set(&gen2, reference_model, 100);

  auto result = ppl::learning::Learning(learning_set)
    .set_max_iterations(2)
    .set_target_accuracy(learning_set.alternatives_count)
    .forbid_using_gpu()
    .set_random_seed(42)
    .set_models_count(5)
    .perform();

  EXPECT_EQ(result.best_model_accuracy, 82);

  auto domain = ppl::Domain<Host>::make(learning_set);
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, result.best_model));
  EXPECT_EQ(ppl::get_accuracy(models, 0), result.best_model_accuracy);
}
