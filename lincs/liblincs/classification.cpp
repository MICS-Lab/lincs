// Copyright 2023-2024 Vincent Jacques

#include "classification.hpp"

#include <iostream>
#include <cassert>

#include "chrones.hpp"
#include "unreachable.hpp"
#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

bool accepted_by_criterion(
  const Problem& problem,
  const Model& model,
  const Alternatives& alternatives,
  const unsigned boundary_index,
  const unsigned alternative_index,
  const unsigned criterion_index
) {
  const auto& performance = alternatives.get_alternatives()[alternative_index].get_profile()[criterion_index];
  const auto& criterion = problem.get_criteria()[criterion_index];
  return dispatch(
    model.get_accepted_values()[criterion_index].get(),
    [&model, &performance, &criterion, boundary_index](const AcceptedValues::RealThresholds& accepted_values) {
      const float value = performance.get_real().get_value();
      const std::optional<float> threshold = accepted_values.get_thresholds()[boundary_index];
      if (threshold) {
        switch (criterion.get_real_values().get_preference_direction()) {
          case Criterion::PreferenceDirection::increasing:
            return value >= *threshold;
          case Criterion::PreferenceDirection::decreasing:
            return value <= *threshold;
          case Criterion::PreferenceDirection::single_peaked:
            assert(false);
        }
        unreachable();
      } else {
        return false;
      }
    },
    [&model, &performance, &criterion, boundary_index](const AcceptedValues::IntegerThresholds& accepted_values) {
      const int value = performance.get_integer().get_value();
      const std::optional<int> threshold = accepted_values.get_thresholds()[boundary_index];
      if (threshold) {
        switch (criterion.get_integer_values().get_preference_direction()) {
          case Criterion::PreferenceDirection::increasing:
            return value >= *threshold;
          case Criterion::PreferenceDirection::decreasing:
            return value <= *threshold;
          case Criterion::PreferenceDirection::single_peaked:
            assert(false);
        }
        unreachable();
      } else {
        return false;
      }
    },
    [&model, &performance, &criterion, boundary_index](const AcceptedValues::EnumeratedThresholds& accepted_values) {
      const auto& ranks = criterion.get_enumerated_values().get_value_ranks();
      const std::string& value = performance.get_enumerated().get_value();
      const std::optional<std::string>& threshold = accepted_values.get_thresholds()[boundary_index];
      if (threshold) {
        return ranks.at(value) >= ranks.at(*threshold);
      } else {
        return false;
      }
    },
    [&model, &performance, &criterion, boundary_index](const AcceptedValues::RealIntervals& accepted_values) -> bool {
      assert(criterion.get_real_values().get_preference_direction() == Criterion::PreferenceDirection::single_peaked);
      const float value = performance.get_real().get_value();
      const auto& interval = accepted_values.get_intervals()[boundary_index];
      if (interval) {
        return value >= interval->first && value <= interval->second;
      } else {
        return false;
      }
    },
    [&model, &performance, &criterion, boundary_index](const AcceptedValues::IntegerIntervals& accepted_values) -> bool {
      assert(criterion.get_integer_values().get_preference_direction() == Criterion::PreferenceDirection::single_peaked);
      const int value = performance.get_integer().get_value();
      const auto& interval = accepted_values.get_intervals()[boundary_index];
      if (interval) {
        return value >= interval->first && value <= interval->second;
      } else {
        return false;
      }
    }
  );
}

bool accepted_by_category(
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
      boost::dynamic_bitset<> accepted_criteria(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (accepted_by_criterion(problem, model, alternatives, boundary_index, alternative_index, criterion_index)) {
          accepted_criteria[criterion_index] = true;
        }
      }
      return sufficient_coalitions.accept(accepted_criteria);
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
      if (accepted_by_category(problem, model, *alternatives, category_index - 1, alternative_index)) {
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

unsigned count_correctly_classified_alternatives(const Problem& problem, const Model& model, const Alternatives& alternatives) {
  CHRONE();

  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned alternatives_count = alternatives.get_alternatives().size();

  unsigned result = 0;

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    unsigned category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      if (accepted_by_category(problem, model, alternatives, category_index - 1, alternative_index)) {
        break;
      }
    }

    if (alternatives.get_alternatives()[alternative_index].get_category_index() == category_index) {
      ++result;
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

TEST_CASE("Classification with unreachable thresholds") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {{"Cat 1"}, {"Cat 2"}, {"Cat 3"}, {"Cat 4"}},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance(Performance::Real(0.24))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.25))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.49))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.50))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.74))}, std::nullopt},
    {"A", {Performance(Performance::Real(0.75))}, std::nullopt},
  }};

  classify_alternatives(
    problem,
    Model{
      problem,
      {
        AcceptedValues(AcceptedValues::RealThresholds({0.25, 0.5, 0.75})),
      },
      {
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
      },
    },
    &alternatives);

  CHECK(*alternatives.get_alternatives()[0].get_category_index() == 0);
  CHECK(*alternatives.get_alternatives()[1].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[2].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[3].get_category_index() == 2);
  CHECK(*alternatives.get_alternatives()[4].get_category_index() == 2);
  CHECK(*alternatives.get_alternatives()[5].get_category_index() == 3);

  classify_alternatives(
    problem,
    Model{
      problem,
      {
        AcceptedValues(AcceptedValues::RealThresholds({0.25, 0.5, std::nullopt})),
      },
      {
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
        SufficientCoalitions(SufficientCoalitions::Weights({1})),
      },
    },
    &alternatives);

  CHECK(*alternatives.get_alternatives()[0].get_category_index() == 0);
  CHECK(*alternatives.get_alternatives()[1].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[2].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[3].get_category_index() == 2);
  CHECK(*alternatives.get_alternatives()[4].get_category_index() == 2);
  CHECK(*alternatives.get_alternatives()[5].get_category_index() == 2);
}

TEST_CASE("Classification with single-peaked criteria") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::single_peaked, 0, 1)),
      Criterion("Criterion 2", Criterion::IntegerValues(Criterion::PreferenceDirection::single_peaked, 0, 100)),
    },
    {
      {"Bad"},
      {"Average"},
      {"Good"},
      {"God-like (unreachable)"}
    },
  };

  Model model{
    problem,
    {
      AcceptedValues(AcceptedValues::RealIntervals({std::make_pair(0.2, 0.8), std::make_pair(0.4, 0.6), std::nullopt})),
      AcceptedValues(AcceptedValues::IntegerIntervals({std::make_pair(20, 80), std::make_pair(40, 60), std::nullopt})),
    },
    {
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
      SufficientCoalitions(SufficientCoalitions::Weights({0.5, 0.5})),
    },
  };

  Alternatives alternatives{problem, {
    {"Good on both", {Performance(Performance::Real(0.5)), Performance(Performance::Integer(50))}, std::nullopt},
    {"Low on 1 => average", {Performance(Performance::Real(0.3)), Performance(Performance::Integer(50))}, std::nullopt},
    {"Low on 2 => average", {Performance(Performance::Real(0.5)), Performance(Performance::Integer(30))}, std::nullopt},
    {"Very low on 1 => bad", {Performance(Performance::Real(0.1)), Performance(Performance::Integer(50))}, std::nullopt},
    {"Very low on 2 => bad", {Performance(Performance::Real(0.5)), Performance(Performance::Integer(10))}, std::nullopt},
    {"High on 1 => average", {Performance(Performance::Real(0.7)), Performance(Performance::Integer(50))}, std::nullopt},
    {"High on 2 => average", {Performance(Performance::Real(0.5)), Performance(Performance::Integer(70))}, std::nullopt},
    {"Very high on 1 => bad", {Performance(Performance::Real(0.9)), Performance(Performance::Integer(50))}, std::nullopt},
    {"Very high on 2 => bad", {Performance(Performance::Real(0.5)), Performance(Performance::Integer(90))}, std::nullopt},
  }};

  classify_alternatives(problem, model, &alternatives);

  CHECK(*alternatives.get_alternatives()[0].get_category_index() == 2);
  CHECK(*alternatives.get_alternatives()[1].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[2].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[3].get_category_index() == 0);
  CHECK(*alternatives.get_alternatives()[4].get_category_index() == 0);
  CHECK(*alternatives.get_alternatives()[5].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[6].get_category_index() == 1);
  CHECK(*alternatives.get_alternatives()[7].get_category_index() == 0);
  CHECK(*alternatives.get_alternatives()[8].get_category_index() == 0);
}

}  // namespace lincs
