// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <gtest/gtest.h>

#include <utility>

#include "cuda-utils.hpp"


TEST(MakeModels, SingleAlternativeSingleCriteria) {
  Domain<Host> domain = Domain<Host>::make(2, {
    {{0.25}, 1},
  });

  EXPECT_EQ(domain.categories_count, 2);
  EXPECT_EQ(domain.criteria_count, 1);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_assignments[0], 1);

  {
    Models<Host> models = Models<Host>::make(domain, {});
    EXPECT_EQ(models.models_count, 0);
  }

  {
    Models<Host> models = Models<Host>::make(domain, {
      {{{0.25}}, {1.}},
    });

    EXPECT_EQ(models.models_count, 1);
    EXPECT_EQ(models.weights[0][0], 1.);
    EXPECT_EQ(models.profiles[0][0][0], 0.25);
  }
}

TEST(MakeModels, SeveralAlternativesSingleCriteria) {
  Domain<Host> domain = Domain<Host>::make(4, {
    {{0.00}, 0},
    {{0.25}, 1},
    {{0.50}, 2},
    {{0.75}, 2},
    {{1.00}, 3},
  });

  EXPECT_EQ(domain.categories_count, 4);
  EXPECT_EQ(domain.criteria_count, 1);
  EXPECT_EQ(domain.learning_alternatives_count, 5);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.00);
  EXPECT_EQ(domain.learning_alternatives[0][1], 0.25);
  EXPECT_EQ(domain.learning_alternatives[0][2], 0.50);
  EXPECT_EQ(domain.learning_alternatives[0][3], 0.75);
  EXPECT_EQ(domain.learning_alternatives[0][4], 1.00);
  EXPECT_EQ(domain.learning_assignments[0], 0);
  EXPECT_EQ(domain.learning_assignments[1], 1);
  EXPECT_EQ(domain.learning_assignments[2], 2);
  EXPECT_EQ(domain.learning_assignments[3], 2);
  EXPECT_EQ(domain.learning_assignments[4], 3);

  Models<Host> models = Models<Host>::make(domain, {
    {{{0.25}, {0.50}, {0.75}}, {1.}},
  });

  EXPECT_EQ(models.models_count, 1);
  EXPECT_EQ(models.weights[0][0], 1.);
  EXPECT_EQ(models.profiles[0][0][0], 0.25);
  EXPECT_EQ(models.profiles[0][1][0], 0.50);
  EXPECT_EQ(models.profiles[0][2][0], 0.75);
}

TEST(MakeModels, SingleAlternativeSeveralCriteria) {
  Domain<Host> domain = Domain<Host>::make(2, {
    {{0.25, 0.75, 0.50}, 1},
  });

  EXPECT_EQ(domain.categories_count, 2);
  EXPECT_EQ(domain.criteria_count, 3);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_alternatives[1][0], 0.75);
  EXPECT_EQ(domain.learning_alternatives[2][0], 0.50);
  EXPECT_EQ(domain.learning_assignments[0], 1);

  Models<Host> models = Models<Host>::make(domain, {
    {{{0.25, 0.50, 0.75}}, {0.25, 0.50, 0.25}},
  });

  EXPECT_EQ(models.models_count, 1);
  EXPECT_EQ(models.weights[0][0], 0.25);
  EXPECT_EQ(models.weights[1][0], 0.50);
  EXPECT_EQ(models.weights[2][0], 0.25);
  EXPECT_EQ(models.profiles[0][0][0], 0.25);
  EXPECT_EQ(models.profiles[1][0][0], 0.50);
  EXPECT_EQ(models.profiles[2][0][0], 0.75);
}

TEST(GetAssignment, SingleCriterion) {
  Domain<Host> domain = Domain<Host>::make(2, {{{0.5}, 0}});

  // Alternative above profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.49}}, {5}}}), 0, 0), 1);
  // Alternative above profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.49}}, {1}}}), 0, 0), 1);
  // Alternative above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.49}}, {0.99}}}), 0, 0), 0);

  // Alternative equal to profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.5}}, {5}}}), 0, 0), 1);
  // Alternative equal to profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.5}}, {1}}}), 0, 0), 1);
  // Alternative equal to profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.5}}, {0.99}}}), 0, 0), 0);

  // Alternative below profile, whatever weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.51}}, {1}}}), 0, 0), 0);
}

TEST(GetAssignment, SeveralCriteria) {
  Domain<Host> domain = Domain<Host>::make(2, {{{0.3, 0.7}, 0}});

  // Alternative fully above profile, heavy weights => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.29, 0.69}}, {5, 5}}}), 0, 0), 1);
  // Alternative above profile on first criterion, heavy weight on first criterion => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.29, 0.71}}, {5, 0.1}}}), 0, 0), 1);
  // Alternative above profile on second criterion, heavy weight on second criterion => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.31, 0.69}}, {0.1, 5}}}), 0, 0), 1);
  // Alternative fully above profile, weights just enough => reach C1
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.29, 0.69}}, {0.5, 0.5}}}), 0, 0), 1);
  // Alternative fully above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.29, 0.69}}, {0.49, 0.49}}}), 0, 0), 0);
  // Alternative above profile on first criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.29, 0.71}}, {0.99, 5}}}), 0, 0), 0);
  // Alternative above profile on second criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(Models<Host>::make(domain, {{{{0.31, 0.69}}, {5, 0.99}}}), 0, 0), 0);
}

TEST(GetAssignmentAndAccuracy, SeveralAlternativesSeveralModels) {
  Domain<Host> domain = Domain<Host>::make(2, {{{0.25}, 0}, {{0.75}, 1}});

  Models<Host> models = Models<Host>::make(domain, {{{{0.9}}, {1}}, {{{0.5}}, {1}}});

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 0);
  EXPECT_EQ(get_accuracy(models, 0), 1);
  EXPECT_EQ(get_assignment(models, 1, 0), 0);
  EXPECT_EQ(get_assignment(models, 1, 1), 1);
  EXPECT_EQ(get_accuracy(models, 1), 2);
}
