// Copyright 2023 Vincent Jacques

#include "classification.hpp"

#include <iostream>
#include <cassert>

#include "chrones.hpp"
#include "unreachable.hpp"
#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

bool better_or_equal(Criterion::CategoryCorrelation category_correlation, float lhs, float rhs) {
  switch (category_correlation) {
    case Criterion::CategoryCorrelation::growing:
      return lhs >= rhs;
    case Criterion::CategoryCorrelation::decreasing:
      return lhs <= rhs;
  }
 unreachable();
}

bool is_good_enough(const Problem& problem, const Model::Boundary& boundary, const Alternative& alternative) {
  const unsigned criteria_count = problem.criteria.size();

  switch (boundary.sufficient_coalitions.kind) {
    case SufficientCoalitions::Kind::weights: {
      float weight_at_or_better_than_profile = 0;
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        const float alternative_value = alternative.profile[criterion_index];
        const float profile_value = boundary.profile[criterion_index];
        if (better_or_equal(problem.criteria[criterion_index].category_correlation, alternative_value, profile_value)) {
          weight_at_or_better_than_profile += boundary.sufficient_coalitions.criterion_weights[criterion_index];
        }
      }
      return weight_at_or_better_than_profile >= 1.f;
    }
    case SufficientCoalitions::Kind::roots: {
      boost::dynamic_bitset<> at_or_better_than_profile(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        const float alternative_value = alternative.profile[criterion_index];
        const float profile_value = boundary.profile[criterion_index];
        if (better_or_equal(problem.criteria[criterion_index].category_correlation, alternative_value, profile_value)) {
          at_or_better_than_profile[criterion_index] = true;
        }
      }

      for (boost::dynamic_bitset<> root: boundary.sufficient_coalitions.upset_roots) {
        if ((at_or_better_than_profile & root) == root) {
          return true;
        }
      }

      return false;
    }
  }
  unreachable();
}

ClassificationResult classify_alternatives(const Problem& problem, const Model& model, Alternatives* alternatives) {
  CHRONE();

  assert(&(model.problem) == &problem);
  assert(&(alternatives->problem) == &problem);

  const unsigned categories_count = problem.categories.size();

  ClassificationResult result{0, 0};

  for (auto& alternative: alternatives->alternatives) {
    unsigned category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      if (is_good_enough(problem, model.boundaries[category_index - 1], alternative)) {
        break;
      }
    }

    if (alternative.category_index == category_index) {
      ++result.unchanged;
    } else {
      alternative.category_index = category_index;
      ++result.changed;
    }
  }

  return result;
}

TEST_CASE("Basic classification using weights") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.5, 0.5, 0.5}, {SufficientCoalitions::weights, {0.3, 0.6, 0.8}}}},
  };

  Alternatives alternatives{problem, {
    {"A", {0.49, 0.49, 0.49}, std::nullopt},
    {"A", {0.50, 0.49, 0.49}, std::nullopt},
    {"A", {0.49, 0.50, 0.49}, std::nullopt},
    {"A", {0.49, 0.49, 0.50}, std::nullopt},
    {"A", {0.49, 0.50, 0.50}, std::nullopt},
    {"A", {0.50, 0.49, 0.50}, std::nullopt},
    {"A", {0.50, 0.50, 0.49}, std::nullopt},
    {"A", {0.50, 0.50, 0.50}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.alternatives[0].category_index == 0);
  CHECK(alternatives.alternatives[1].category_index == 0);
  CHECK(alternatives.alternatives[2].category_index == 0);
  CHECK(alternatives.alternatives[3].category_index == 0);
  CHECK(alternatives.alternatives[4].category_index == 1);
  CHECK(alternatives.alternatives[5].category_index == 1);
  CHECK(alternatives.alternatives[6].category_index == 0);
  CHECK(alternatives.alternatives[7].category_index == 1);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

TEST_CASE("Basic classification using upset roots") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::growing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.5, 0.5, 0.5}, {SufficientCoalitions::roots, 3, {{0, 2}, {1, 2}}}}},
  };

  Alternatives alternatives{problem, {
    {"A", {0.49, 0.49, 0.49}, std::nullopt},
    {"A", {0.50, 0.49, 0.49}, std::nullopt},
    {"A", {0.49, 0.50, 0.49}, std::nullopt},
    {"A", {0.49, 0.49, 0.50}, std::nullopt},
    {"A", {0.49, 0.50, 0.50}, std::nullopt},
    {"A", {0.50, 0.49, 0.50}, std::nullopt},
    {"A", {0.50, 0.50, 0.49}, std::nullopt},
    {"A", {0.50, 0.50, 0.50}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.alternatives[0].category_index == 0);
  CHECK(alternatives.alternatives[1].category_index == 0);
  CHECK(alternatives.alternatives[2].category_index == 0);
  CHECK(alternatives.alternatives[3].category_index == 0);
  CHECK(alternatives.alternatives[4].category_index == 1);
  CHECK(alternatives.alternatives[5].category_index == 1);
  CHECK(alternatives.alternatives[6].category_index == 0);
  CHECK(alternatives.alternatives[7].category_index == 1);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

TEST_CASE("Classification with decreasing criteria") {
  Problem problem{
    {
      {"Criterion 1", Criterion::ValueType::real, Criterion::CategoryCorrelation::decreasing, 0, 1},
      {"Criterion 2", Criterion::ValueType::real, Criterion::CategoryCorrelation::decreasing, 0, 1},
      {"Criterion 3", Criterion::ValueType::real, Criterion::CategoryCorrelation::decreasing, 0, 1},
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {{{0.5, 0.5, 0.5}, {SufficientCoalitions::weights, {0.3, 0.6, 0.8}}}},
  };

  Alternatives alternatives{problem, {
    {"A", {0.50, 0.50, 0.50}, std::nullopt},
    {"A", {0.51, 0.50, 0.50}, std::nullopt},
    {"A", {0.50, 0.51, 0.50}, std::nullopt},
    {"A", {0.50, 0.50, 0.51}, std::nullopt},
    {"A", {0.50, 0.51, 0.51}, std::nullopt},
    {"A", {0.51, 0.50, 0.51}, std::nullopt},
    {"A", {0.51, 0.51, 0.50}, std::nullopt},
    {"A", {0.51, 0.51, 0.51}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.alternatives[0].category_index == 1);
  CHECK(alternatives.alternatives[1].category_index == 1);
  CHECK(alternatives.alternatives[2].category_index == 1);
  CHECK(alternatives.alternatives[3].category_index == 0);
  CHECK(alternatives.alternatives[4].category_index == 0);
  CHECK(alternatives.alternatives[5].category_index == 0);
  CHECK(alternatives.alternatives[6].category_index == 0);
  CHECK(alternatives.alternatives[7].category_index == 0);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

}  // namespace lincs
