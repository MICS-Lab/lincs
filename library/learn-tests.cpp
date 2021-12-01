// Copyright 2021 Vincent Jacques

#include "assign.hpp"
#include "generate.hpp"
#include "learn.hpp"
#include "test-utils.hpp"


TEST(Learn, First) {
  std::mt19937 gen1(42);
  auto reference_model = ppl::generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = ppl::generate::learning_set(&gen2, reference_model, 100);

  RandomSource random;
  random.init_for_host(42);

  auto reconstructed_model = ppl::learn::learn_from(random, learning_set);

  auto domain = ppl::Domain<Host>::make(learning_set);
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, reconstructed_model));

  EXPECT_GE(ppl::get_accuracy(models, 0), 98);
}
