// Copyright 2021-2022 Vincent Jacques

#include <chrones.hpp>

#include "../randomness.hpp"
#include "../test-utils.hpp"


CHRONABLE("max-power-per-criterion-tests")

namespace ppl {

// Internal function (not declared in the header) that we still want to unit-test
std::map<float, double> get_candidate_probabilities(const DomainView&, uint crit_index, uint profile_index);

std::map<float, double> normalize(const std::map<float, double>& value_probabilities) {
  return ProbabilityWeightedGenerator<float>::make(value_probabilities).get_value_probabilities();
}

std::map<float, double> get_candidates(const Domain<Host>& domain, uint crit_index, uint profile_index) {
  return normalize(get_candidate_probabilities(domain.get_view(), crit_index, profile_index));
}

TEST(GetValueProbabilities, OneAlternativeBelow) {
  auto domain = make_domain(2, {{{0.5}, 0}});

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({{0.5, 1}}));
}

TEST(GetValueProbabilities, FourAlternativesBelow) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.4}, 0},
    {{0.5}, 0},
    {{0.6}, 0},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy one category is empty)
      {0.3, 0},
      {0.4, 1},
      {0.5, 2},
      {0.6, 3},
    }));
}

TEST(GetValueProbabilities, OneAlternativeAbove) {
  auto domain = make_domain(2, {{{0.5}, 1}});

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({{0.5, 1}}));
}

TEST(GetValueProbabilities, FourAlternativesAbove) {
  auto domain = make_domain(2, {
    {{0.3}, 1},
    {{0.4}, 1},
    {{0.5}, 1},
    {{0.6}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy one category is empty)
      {0.3, 4},
      {0.4, 3},
      {0.5, 2},
      {0.6, 1},
    }));
}

TEST(GetValueProbabilities, TwoAlternativesWellSplit) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.7}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy because categories have same size)
      {0.3, 1},
      {0.7, 2},
    }));
}

TEST(GetValueProbabilities, TwoAlternativesInverted) {
  auto domain = make_domain(2, {
    {{0.3}, 1},
    {{0.7}, 0},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy because categories have same size)
      {0.3, 1},
      {0.7, 0},
    }));
}

TEST(GetValueProbabilities, FourAlternativesWellSplit) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.4}, 0},
    {{0.6}, 1},
    {{0.7}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy because categories have same size)
      {0.3, 2},
      {0.4, 3},
      {0.6, 4},
      {0.7, 3},
    }));
}

TEST(GetValueProbabilities, FourAlternativesWithRepeatedValuesWellSplit) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.3}, 0},
    {{0.7}, 1},
    {{0.7}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy because categories have same size)
      {0.3, 2},
      {0.7, 4},
    }));
}

TEST(GetValueProbabilities, FourAlternativesMixed) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.4}, 1},
    {{0.6}, 0},
    {{0.7}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy because categories have same size)
      {0.3, 2},
      {0.4, 3},
      {0.6, 2},
      {0.7, 3},
    }));
}

TEST(GetValueProbabilities, ThreeAlternativesWellSplit) {
  auto domain = make_domain(2, {
    {{0.3}, 0},
    {{0.5}, 0},
    {{0.7}, 1},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy / size of category (general rule))
      {0.3, 1. / 2},
      {0.5, 2. / 2},
      {0.7, 3. / 1},
    }));
}

TEST(GetValueProbabilities, ThreeAlternativesInverted) {
  auto domain = make_domain(2, {
    {{0.3}, 1},
    {{0.5}, 1},
    {{0.7}, 0},
  });

  EXPECT_EQ(
    get_candidates(domain, 0, 0),
    normalize({
      // First column: profile location
      // Second column: probability (== accuracy / size of category (general rule))
      {0.3, 2},
      {0.5, 1},
      {0.7, 0},
    }));
}

}  // namespace ppl
