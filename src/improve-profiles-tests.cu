// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <gtest/gtest.h>

#include <utility>

#include "cuda-utils.hpp"


TEST(MakeDomain, SingleAlternativeSingleCriteria) {
  Domain<Host> domain = Domain<Host>::make({
    {{0.25}, 0},
  });

  EXPECT_EQ(domain.categories_count, 1);
  EXPECT_EQ(domain.criteria_count, 1);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_assignments[0], 0);
}

TEST(MakeDomain, SeveralAlternativesSingleCriteria) {
  Domain<Host> domain = Domain<Host>::make({
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
}

TEST(MakeDomain, SingleAlternativeSeveralCriteria) {
  Domain<Host> domain = Domain<Host>::make({
    {{0.25, 0.75, 0.50}, 0},
  });

  EXPECT_EQ(domain.categories_count, 1);
  EXPECT_EQ(domain.criteria_count, 3);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_alternatives[1][0], 0.75);
  EXPECT_EQ(domain.learning_alternatives[2][0], 0.50);
  EXPECT_EQ(domain.learning_assignments[0], 0);
}
