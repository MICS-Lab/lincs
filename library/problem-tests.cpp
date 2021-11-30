// Copyright 2021 Vincent Jacques

#include <gtest/gtest.h>

#include "problem.hpp"


using ppl::Domain;
using ppl::Models;

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


TEST(MakeModels, SingleAlternativeSingleCriteria) {
  auto domain = make_domain(2, {
    {{0.25}, 1},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 2);
  EXPECT_EQ(domain_view.criteria_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain_view.learning_assignments[0], 1);

  {
    auto models = make_models(domain, {});
    auto models_view = models.get_view();

    EXPECT_EQ(models_view.models_count, 0);
    EXPECT_EQ(models_view.weights.s1(), 1);
    EXPECT_EQ(models_view.weights.s0(), 0);
    EXPECT_EQ(models_view.profiles.s2(), 1);
    EXPECT_EQ(models_view.profiles.s1(), 1);
    EXPECT_EQ(models_view.profiles.s0(), 0);
  }

  {
    auto models = make_models(domain, {
      {{{0.25}}, {1.}},
    });
    auto models_view = models.get_view();

    EXPECT_EQ(models_view.models_count, 1);
    EXPECT_EQ(models_view.weights.s1(), 1);
    EXPECT_EQ(models_view.weights.s0(), 1);
    EXPECT_EQ(models_view.weights[0][0], 1.);
    EXPECT_EQ(models_view.profiles.s2(), 1);
    EXPECT_EQ(models_view.profiles.s1(), 1);
    EXPECT_EQ(models_view.profiles.s0(), 1);
    EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  }
}

TEST(MakeModels, SeveralAlternativesSingleCriteria) {
  auto domain = make_domain(4, {
    {{0.00}, 0},
    {{0.25}, 1},
    {{0.50}, 2},
    {{0.75}, 2},
    {{1.00}, 3},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 4);
  EXPECT_EQ(domain_view.criteria_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives_count, 5);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.00);
  EXPECT_EQ(domain_view.learning_alternatives[0][1], 0.25);
  EXPECT_EQ(domain_view.learning_alternatives[0][2], 0.50);
  EXPECT_EQ(domain_view.learning_alternatives[0][3], 0.75);
  EXPECT_EQ(domain_view.learning_alternatives[0][4], 1.00);
  EXPECT_EQ(domain_view.learning_assignments[0], 0);
  EXPECT_EQ(domain_view.learning_assignments[1], 1);
  EXPECT_EQ(domain_view.learning_assignments[2], 2);
  EXPECT_EQ(domain_view.learning_assignments[3], 2);
  EXPECT_EQ(domain_view.learning_assignments[4], 3);

  auto models = make_models(domain, {
    {{{0.25}, {0.50}, {0.75}}, {1.}},
  });
  auto models_view = models.get_view();

  EXPECT_EQ(models_view.models_count, 1);
  EXPECT_EQ(models_view.weights[0][0], 1.);
  EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  EXPECT_EQ(models_view.profiles[0][1][0], 0.50);
  EXPECT_EQ(models_view.profiles[0][2][0], 0.75);
}

TEST(MakeModels, SingleAlternativeSeveralCriteria) {
  auto domain = make_domain(2, {
    {{0.25, 0.75, 0.50}, 1},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 2);
  EXPECT_EQ(domain_view.criteria_count, 3);
  EXPECT_EQ(domain_view.learning_alternatives_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain_view.learning_alternatives[1][0], 0.75);
  EXPECT_EQ(domain_view.learning_alternatives[2][0], 0.50);
  EXPECT_EQ(domain_view.learning_assignments[0], 1);

  auto models = make_models(domain, {
    {{{0.25, 0.50, 0.75}}, {0.25, 0.50, 0.25}},
  });
  auto models_view = models.get_view();

  EXPECT_EQ(models_view.models_count, 1);
  EXPECT_EQ(models_view.weights[0][0], 0.25);
  EXPECT_EQ(models_view.weights[1][0], 0.50);
  EXPECT_EQ(models_view.weights[2][0], 0.25);
  EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  EXPECT_EQ(models_view.profiles[1][0][0], 0.50);
  EXPECT_EQ(models_view.profiles[2][0][0], 0.75);
}
