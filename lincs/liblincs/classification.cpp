// Copyright 2023 Vincent Jacques

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
  const auto& performance = alternatives.alternatives[alternative_index].profile[criterion_index];
  const auto& accepted_values = model.accepted_values[criterion_index];
  return dispatch(
    problem.criteria[criterion_index].get_values(),
    [&model, &performance, &accepted_values, boundary_index](const Criterion::RealValues& values) {
      const float threshold = accepted_values.get_real_thresholds()[boundary_index];
      return better_or_equal(values.preference_direction, performance.get_real_value(), threshold);
    },
    [&model, &performance, &accepted_values, boundary_index](const Criterion::IntegerValues& values) {
      const int threshold = accepted_values.get_integer_thresholds()[boundary_index];
      return better_or_equal(values.preference_direction, performance.get_integer_value(), threshold);;
    },
    [&model, &performance, &accepted_values, boundary_index](const Criterion::EnumeratedValues& values) {
      const std::string threshold_enum = accepted_values.get_enumerated_thresholds()[boundary_index];
      return values.value_ranks.at(performance.get_enumerated_value()) >= values.value_ranks.at(threshold_enum);
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
  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.ordered_categories.size();
  const unsigned boundaries_count = categories_count - 1;

  assert(model.accepted_values.size() == criteria_count);
  assert(model.sufficient_coalitions.size() == boundaries_count);
  assert(boundary_index < boundaries_count);

  switch (model.sufficient_coalitions[boundary_index].get_kind()) {
    case SufficientCoalitions::Kind::weights: {
      float weight_at_or_better_than_profile = 0;
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (better_or_equal(problem, model, alternatives, boundary_index, alternative_index, criterion_index)) {
          weight_at_or_better_than_profile += model.sufficient_coalitions[boundary_index].get_criterion_weights()[criterion_index];
        }
      }
      return weight_at_or_better_than_profile >= 1.f;
    }
    case SufficientCoalitions::Kind::roots: {
      boost::dynamic_bitset<> at_or_better_than_profile(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (better_or_equal(problem, model, alternatives, boundary_index, alternative_index, criterion_index)) {
          at_or_better_than_profile[criterion_index] = true;
        }
      }

      for (boost::dynamic_bitset<> root: model.sufficient_coalitions[boundary_index].get_upset_roots_as_bitsets()) {
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

  const unsigned categories_count = problem.ordered_categories.size();
  const unsigned alternatives_count = alternatives->alternatives.size();

  ClassificationResult result{0, 0};

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    unsigned category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      if (is_good_enough(problem, model, *alternatives, category_index - 1, alternative_index)) {
        break;
      }
    }

    if (alternatives->alternatives[alternative_index].category_index == category_index) {
      ++result.unchanged;
    } else {
      alternatives->alternatives[alternative_index].category_index = category_index;
      ++result.changed;
    }
  }

  return result;
}

TEST_CASE("Basic classification using weights") {
  Problem problem{
    {
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
    },
    {SufficientCoalitions::make_weights({0.3, 0.6, 0.8})},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance::make_real(0.49), Performance::make_real(0.49), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.49), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.50), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.49), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.49), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
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
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::increasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::increasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
    },
    {SufficientCoalitions::make_roots_from_vectors(3, {{0, 2}, {1, 2}})},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance::make_real(0.49), Performance::make_real(0.49), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.49), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.50), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.49), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.49), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.49), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.49)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
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
      Criterion::make_real("Criterion 1", Criterion::PreferenceDirection::decreasing, 0, 1),
      Criterion::make_real("Criterion 2", Criterion::PreferenceDirection::decreasing, 0, 1),
      Criterion::make_real("Criterion 3", Criterion::PreferenceDirection::decreasing, 0, 1),
    },
    {{"Category 1"}, {"Category 2"}},
  };

  Model model{
    problem,
    {
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
      AcceptedValues::make_real_thresholds({0.5}),
    },
    {SufficientCoalitions::make_weights({0.3, 0.6, 0.8})},
  };

  Alternatives alternatives{problem, {
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.51), Performance::make_real(0.50), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.51), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.50), Performance::make_real(0.51)}, std::nullopt},
    {"A", {Performance::make_real(0.50), Performance::make_real(0.51), Performance::make_real(0.51)}, std::nullopt},
    {"A", {Performance::make_real(0.51), Performance::make_real(0.50), Performance::make_real(0.51)}, std::nullopt},
    {"A", {Performance::make_real(0.51), Performance::make_real(0.51), Performance::make_real(0.50)}, std::nullopt},
    {"A", {Performance::make_real(0.51), Performance::make_real(0.51), Performance::make_real(0.51)}, std::nullopt},
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
