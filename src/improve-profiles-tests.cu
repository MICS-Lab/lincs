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
