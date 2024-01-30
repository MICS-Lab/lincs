// Copyright 2023-2024 Vincent Jacques

#include "classification.hpp"

#include <iostream>
#include <cassert>

#include "chrones.hpp"
#include "unreachable.hpp"
#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

bool better_or_equal(
  const Problem& problem,
  const Model& model,
  const Alternatives& alternatives,
  const unsigned boundary_index,
  const unsigned alternative_index,
  const unsigned criterion_index
) {
  const auto& performance = alternatives.get_alternatives()[alternative_index].get_profile()[criterion_index];
  const auto& accepted_values = model.get_accepted_values()[criterion_index];
  return dispatch(
    problem.get_criteria()[criterion_index].get_values(),
    [&model, &performance, &accepted_values, boundary_index](const Criterion::RealValues& values) {
      const float threshold = accepted_values.get_real_thresholds().get_thresholds()[boundary_index];
      return better_or_equal(values.get_preference_direction(), performance.get_real().get_value(), threshold);
    },
    [&model, &performance, &accepted_values, boundary_index](const Criterion::IntegerValues& values) {
      const int threshold = accepted_values.get_integer_thresholds().get_thresholds()[boundary_index];
      return better_or_equal(values.get_preference_direction(), performance.get_integer().get_value(), threshold);;
    },
    [&model, &performance, &accepted_values, boundary_index](const Criterion::EnumeratedValues& values) {
      const std::string threshold_enum = accepted_values.get_enumerated_thresholds().get_thresholds()[boundary_index];
      return values.get_value_ranks().at(performance.get_enumerated().get_value()) >= values.get_value_ranks().at(threshold_enum);
    }
  );
}

bool is_good_enough(
  const Problem& problem,
  const Model& model,
  const Alternatives& alternatives,
  const unsigned boundary_index,
  const unsigned alternative_index
) {
  const unsigned criteria_count = problem.get_criteria().size();
  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned boundaries_count = categories_count - 1;

  assert(model.get_accepted_values().size() == criteria_count);
  assert(model.get_sufficient_coalitions().size() == boundaries_count);
  assert(boundary_index < boundaries_count);

  return std::visit(
    [&problem, &model, &alternatives, criteria_count, boundary_index, alternative_index](const auto& sufficient_coalitions) {
      boost::dynamic_bitset<> at_or_better_than_profile(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (better_or_equal(problem, model, alternatives, boundary_index, alternative_index, criterion_index)) {
          at_or_better_than_profile[criterion_index] = true;
        }
      }
      return sufficient_coalitions.accept(at_or_better_than_profile);
    },
    model.get_sufficient_coalitions()[boundary_index].get()
  );
}

ClassificationResult classify_alternatives(const Problem& problem, const Model& model, Alternatives* alternatives) {
  CHRONE();

  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned alternatives_count = alternatives->get_alternatives().size();

  ClassificationResult result{0, 0};

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    unsigned category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      if (is_good_enough(problem, model, *alternatives, category_index - 1, alternative_index)) {
        break;
      }
    }

    if (alternatives->get_alternatives()[alternative_index].get_category_index() == category_index) {
      ++result.unchanged;
    } else {
      alternatives->get_writable_alternatives()[alternative_index].set_category_index(category_index);
      ++result.changed;
    }
  }

  return result;
}

TEST_CASE("Basic classification using weights") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
    },
    {SufficientCoalitions(SufficientCoalitions::Weights({0.3, 0.6, 0.8}))},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.get_alternatives()[0].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[1].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[2].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[3].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[4].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[5].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[6].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[7].get_category_index() == 1);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

TEST_CASE("Basic classification using upset roots") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
    },
    {SufficientCoalitions(SufficientCoalitions::Roots(problem, {{0, 2}, {1, 2}}))},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.get_alternatives()[0].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[1].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[2].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[3].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[4].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[5].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[6].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[7].get_category_index() == 1);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

TEST_CASE("Classification with decreasing criteria") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::decreasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::decreasing, 0, 1)),
      Criterion("Criterion 3", Criterion::RealValues(Criterion::PreferenceDirection::decreasing, 0, 1)),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
      AcceptedValues(AcceptedValues::RealThresholds({0.5})),
    },
    {SufficientCoalitions(SufficientCoalitions::Weights({0.3, 0.6, 0.8}))},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.51)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.51)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.51))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50)), Performance(Performance::Real(0.51)), Performance(Performance::Real(0.51))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.51)), Performance(Performance::Real(0.50)), Performance(Performance::Real(0.51))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.51)), Performance(Performance::Real(0.51)), Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.51)), Performance(Performance::Real(0.51)), Performance(Performance::Real(0.51))}, std::nullopt},
  }};

  auto result = classify_alternatives(problem, model, &alternatives);

  CHECK(alternatives.get_alternatives()[0].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[1].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[2].get_category_index() == 1);
  CHECK(alternatives.get_alternatives()[3].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[4].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[5].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[6].get_category_index() == 0);
  CHECK(alternatives.get_alternatives()[7].get_category_index() == 0);
  CHECK(result.unchanged == 0);
  CHECK(result.changed == 8);
}

}  // namespace lincs
