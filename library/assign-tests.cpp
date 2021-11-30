// Copyright 2021 Vincent Jacques

#include <gtest/gtest.h>

#include "generate.hpp"
#include "assign.hpp"


using ppl::Domain;
using ppl::Models;
using ppl::get_assignment;
using ppl::get_accuracy;

Domain<Host> make_domain(
  const uint categories_count,
  const std::vector<std::pair<std::vector<float>, uint>>& alternatives_
) {
  const uint criteria_count = alternatives_.front().first.size();
  const uint alternatives_count = alternatives_.size();

  std::vector<ppl::io::ClassifiedAlternative> alternatives;
  for (auto alternative : alternatives_) {
    alternatives.push_back(ppl::io::ClassifiedAlternative(alternative.first, alternative.second));
  }

  return Domain<Host>::make(ppl::io::LearningSet(criteria_count, categories_count, alternatives_count, alternatives));
}

Models<Host> make_models(
  const Domain<Host>& domain,
  const std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>& models_
) {
  const uint criteria_count = domain.get_view().criteria_count;
  const uint categories_count = domain.get_view().categories_count;
  const uint models_count = models_.size();

  std::vector<ppl::io::Model> models;
  for (auto model : models_) {
    models.push_back(ppl::io::Model(criteria_count, categories_count, model.first, model.second));
  }

  return Models<Host>::make(domain, models);
}


TEST(GetAssignment, SingleCriterion) {
  auto domain = make_domain(2, {{{0.5}, 0}});

  // Alternative above profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {5}}}), 0, 0), 1);
  // Alternative above profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {1}}}), 0, 0), 1);
  // Alternative above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {0.99}}}), 0, 0), 0);

  // Alternative equal to profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {5}}}), 0, 0), 1);
  // Alternative equal to profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {1}}}), 0, 0), 1);
  // Alternative equal to profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {0.99}}}), 0, 0), 0);

  // Alternative below profile, whatever weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.51}}, {1}}}), 0, 0), 0);
}

TEST(GetAssignment, SingleCriterionManyCategories) {
  auto domain = make_domain(4, {{{0.2}, 0}, {{0.4}, 1}, {{0.6}, 2}, {{0.8}, 3}});
  auto models = make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}});

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 1);
  EXPECT_EQ(get_assignment(models, 0, 2), 2);
  EXPECT_EQ(get_assignment(models, 0, 3), 3);
}

TEST(GetAssignment, SeveralCriteria) {
  auto domain = make_domain(2, {{{0.3, 0.7}, 0}});

  // Alternative fully above profile, heavy weights => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {5, 5}}}), 0, 0), 1);
  // Alternative above profile on first criterion, heavy weight on first criterion => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.71}}, {5, 0.1}}}), 0, 0), 1);
  // Alternative above profile on second criterion, heavy weight on second criterion => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.31, 0.69}}, {0.1, 5}}}), 0, 0), 1);
  // Alternative fully above profile, weights just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {0.5, 0.5}}}), 0, 0), 1);
  // Alternative fully above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {0.49, 0.49}}}), 0, 0), 0);
  // Alternative above profile on first criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.71}}, {0.99, 5}}}), 0, 0), 0);
  // Alternative above profile on second criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.31, 0.69}}, {5, 0.99}}}), 0, 0), 0);
}

TEST(GetAssignmentAndAccuracy, SeveralAlternativesSeveralModels) {
  auto domain = make_domain(2, {{{0.25}, 0}, {{0.75}, 1}});
  auto device_domain = domain.clone_to<Device>();

  auto models = make_models(domain, {{{{0.9}}, {1}}, {{{0.5}}, {1}}});
  auto device_models = models.clone_to<Device>(device_domain);

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 0);
  EXPECT_EQ(get_accuracy(models, 0), 1);
  EXPECT_EQ(get_accuracy(device_models, 0), 1);
  EXPECT_EQ(get_assignment(models, 1, 0), 0);
  EXPECT_EQ(get_assignment(models, 1, 1), 1);
  EXPECT_EQ(get_accuracy(models, 1), 2);
  EXPECT_EQ(get_accuracy(device_models, 1), 2);
}

TEST(GetAccuracy, RandomDomainUniformModel) {
  std::mt19937 gen1(57);
  auto reference_model = ppl::generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = ppl::generate::learning_set(&gen2, reference_model, 100'000);

  auto model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 2., learning_set.categories_count);
  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, std::vector<ppl::io::Model>(1, model));

  const unsigned int expected_accuracy = 44'667;

  #ifndef NOSTOPWATCH
  for (int i = 0; i != 10; ++i)
  #endif
  ASSERT_EQ(get_accuracy(models, 0), expected_accuracy);

  auto device_domain = domain.clone_to<Device>();
  auto device_models = models.clone_to<Device>(device_domain);

  #ifndef NOSTOPWATCH
  for (int i = 0; i != 10; ++i)
  #endif
  ASSERT_EQ(get_accuracy(device_models, 0), expected_accuracy);
}
