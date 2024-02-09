// Copyright 2023-2024 Vincent Jacques

#include "generation.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>
#include <set>

#include "chrones.hpp"
#include "classification.hpp"

#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace {

bool env_is_true(const char* name) {
  const char* value = std::getenv(name);
  return value && std::string(value) == "true";
}

const bool skip_long = env_is_true("LINCS_DEV_SKIP_LONG");

}  // namespace

namespace lincs {

Problem generate_classification_problem(
  const unsigned criteria_count,
  const unsigned categories_count,
  const unsigned random_seed,
  bool normalized_min_max,
  const std::vector<Criterion::PreferenceDirection>& allowed_preference_directions,
  const std::vector<Criterion::ValueType>& allowed_value_types
) {
  CHRONE();

  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> min_max_distribution(-100, 100);
  assert(allowed_value_types.size() > 0);
  std::uniform_int_distribution<unsigned> value_type_distribution(0, allowed_value_types.size() - 1);
  assert(allowed_preference_directions.size() > 0);
  std::uniform_int_distribution<unsigned> preference_direction_distribution(0, allowed_preference_directions.size() - 1);

  // Hopping through hoops to generate the same problem as in previous versions:
  //  - first call the RNG for min, max, and direction for each criterion
  //  - then call the RNG for value type for each criterion
  // @todo(Project management, later) Re-explore my old idea of a tree of RNGs for procedural generation
  std::vector<std::tuple<std::string, float, float, Criterion::PreferenceDirection>> criteria_data;
  criteria_data.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    float min_value = 0;
    float max_value = 1;
    if (!normalized_min_max) {
      min_value = min_max_distribution(gen);
      max_value = min_max_distribution(gen);
      if (min_value > max_value) {
        std::swap(min_value, max_value);
      }
    }

    const std::string name = "Criterion " + std::to_string(criterion_index + 1);
    const Criterion::PreferenceDirection direction = allowed_preference_directions[preference_direction_distribution(gen)];
    criteria_data.emplace_back(name, min_value, max_value, direction);
  }

  std::vector<Criterion> criteria;
  criteria.reserve(criteria_count);
  for (const auto [name, min_value, max_value, direction] : criteria_data) {
    switch (allowed_value_types[value_type_distribution(gen)]) {
      case Criterion::ValueType::real:
        criteria.emplace_back(Criterion(name, Criterion::RealValues(direction, min_value, max_value)));
        break;
      case Criterion::ValueType::integer:
        criteria.emplace_back(Criterion(name, Criterion::IntegerValues(direction, 100 * min_value, 100 * max_value)));
        break;
      case Criterion::ValueType::enumerated:
        const unsigned values_count = std::uniform_int_distribution<unsigned>(2, 10)(gen);
        std::vector<std::string> values;
        std::set<std::string> unique_values;
        values.reserve(values_count);
        while (values.size() < values_count) {
          const std::string value({
            "bgrtpd"[std::uniform_int_distribution<unsigned>(0, 5)(gen)],
            "aeiou"[std::uniform_int_distribution<unsigned>(0, 4)(gen)],
            "zrt"[std::uniform_int_distribution<unsigned>(0, 2)(gen)],
          });
          if (unique_values.insert(value).second) {
            values.push_back(value);
          }
        }
        criteria.emplace_back(Criterion(name, Criterion::EnumeratedValues(values)));
        break;
    }
  }

  std::vector<Category> categories;
  categories.reserve(categories_count);
  categories.emplace_back("Worst category");
  assert(categories_count >= 2);
  for (unsigned category_index = 1; category_index < categories_count - 1; ++category_index) {
    categories.emplace_back("Intermediate category " + std::to_string(category_index));
  }
  categories.emplace_back("Best category");

  return Problem{criteria, categories};
}

Model generate_mrsort_classification_model(const Problem& problem, const unsigned random_seed, const std::optional<float> fixed_weights_sum) {
  CHRONE();

  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned boundaries_count = categories_count - 1;
  const unsigned criteria_count = problem.get_criteria().size();

  std::mt19937 gen(random_seed);

  typedef std::variant<float, int, std::string> Performance;
  std::vector<std::vector<Performance>> profiles(boundaries_count, std::vector<Performance>(criteria_count));
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [&gen, boundaries_count, &profiles, criterion_index](const Criterion::RealValues& values) {
        // Profile can take any values. We arbitrarily generate them uniformly
        std::uniform_real_distribution<float> values_distribution(values.get_min_value() + 0.01f, values.get_max_value() - 0.01f);
        // (Values clamped strictly inside ']min, max[' to make it easier to generate balanced learning sets)
        // Profiles must be ordered on each criterion, so we generate a random column...
        std::vector<float> column(boundaries_count);
        std::generate(
          column.begin(), column.end(),
          [&values_distribution, &gen]() { return values_distribution(gen); });
        // ... sort it according to the criterion's preference direction...
        std::sort(column.begin(), column.end(), [&values](float left, float right) { return better_or_equal(values.get_preference_direction(), right, left); });
        // ... and assign that column across all profiles.
        for (unsigned profile_index = 0; profile_index != boundaries_count; ++profile_index) {
          profiles[profile_index][criterion_index] = column[profile_index];
        }
      },
      [&gen, boundaries_count, &profiles, criterion_index](const Criterion::IntegerValues& values) {
        std::uniform_int_distribution<int> values_distribution(values.get_min_value(), values.get_max_value());
        std::vector<int> column(boundaries_count);
        std::generate(
          column.begin(), column.end(),
          [&values_distribution, &gen]() { return values_distribution(gen); });
        std::sort(column.begin(), column.end(), [&values](float left, float right) { return better_or_equal(values.get_preference_direction(), right, left); });
        for (unsigned profile_index = 0; profile_index != boundaries_count; ++profile_index) {
          profiles[profile_index][criterion_index] = column[profile_index];
        }
      },
      [&gen, boundaries_count, &profiles, criterion_index](const Criterion::EnumeratedValues& values) {
        std::uniform_int_distribution<unsigned> values_distribution(0, values.get_ordered_values().size() - 1);
        std::vector<unsigned> ranks(boundaries_count);
        std::generate(
          ranks.begin(), ranks.end(),
          [&values_distribution, &gen]() { return values_distribution(gen); });
        std::sort(ranks.begin(), ranks.end(), [](unsigned left, unsigned right) { return right >= left; });
        for (unsigned profile_index = 0; profile_index != boundaries_count; ++profile_index) {
          profiles[profile_index][criterion_index] = values.get_ordered_values()[ranks[profile_index]];
        }
      }
    );
  }

  // Weights are a bit trickier.
  // We first generate partial sums of weights...
  std::uniform_real_distribution<float> partial_sums_distribution(0.0f, 1.f);
  std::vector<float> partial_sums(criteria_count + 1);
  partial_sums[0] = 0;  // First partial sum is zero
  std::generate(
    std::next(partial_sums.begin()), std::prev(partial_sums.end()),
    [&partial_sums_distribution, &gen]() { return partial_sums_distribution(gen); });
  partial_sums[criteria_count] = 1;  // Last partial sum is one
  // ... sort them...
  std::sort(partial_sums.begin(), partial_sums.end());
  // ... and use consecutive differences as (normalized) weights
  std::vector<float> normalized_weights(criteria_count);
  std::transform(
    partial_sums.begin(), std::prev(partial_sums.end()),
    std::next(partial_sums.begin()),
    normalized_weights.begin(),
    [](float left, float right) { return right - left; });
  // Finally, we denormalize weights so that they add up to a pseudo-random value not less than 1
  assert(!fixed_weights_sum || *fixed_weights_sum >= 1);
  const float weights_sum = fixed_weights_sum ? *fixed_weights_sum : 1.f / std::uniform_real_distribution<float>(0.f, 1.f)(gen);
  std::vector<float> denormalized_weights(criteria_count);
  std::transform(
    normalized_weights.begin(), normalized_weights.end(),
    denormalized_weights.begin(),
    [weights_sum](float w) { return w * weights_sum; });

  std::vector<AcceptedValues> accepted_values;
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    accepted_values.push_back(dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [boundaries_count, &profiles, criterion_index](const Criterion::RealValues&) {
          std::vector<float> thresholds;
          thresholds.reserve(boundaries_count);
          for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
            thresholds.push_back(std::get<float>(profiles[boundary_index][criterion_index]));
          }
          return AcceptedValues(AcceptedValues::RealThresholds(thresholds));
      },
      [boundaries_count, &profiles, criterion_index](const Criterion::IntegerValues&) {
          std::vector<int> thresholds;
          thresholds.reserve(boundaries_count);
          for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
            thresholds.push_back(std::get<int>(profiles[boundary_index][criterion_index]));
          }
          return AcceptedValues(AcceptedValues::IntegerThresholds(thresholds));
      },
      [boundaries_count, &profiles, criterion_index](const Criterion::EnumeratedValues&) {
          std::vector<std::string> thresholds;
          thresholds.reserve(boundaries_count);
          for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
            thresholds.push_back(std::get<std::string>(profiles[boundary_index][criterion_index]));
          }
          return AcceptedValues(AcceptedValues::EnumeratedThresholds(thresholds));
      }
    ));
  }

  std::vector<SufficientCoalitions> sufficient_coalitions;
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    sufficient_coalitions.emplace_back(SufficientCoalitions(SufficientCoalitions::Weights(denormalized_weights)));
  }

  return Model(problem, accepted_values, sufficient_coalitions);
}

TEST_CASE("Generate MR-Sort model - random weights sum") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42);

  std::ostringstream oss;
  model.dump(problem, oss);
  CHECK(oss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.377049327]
  - kind: thresholds
    thresholds: [0.790612161]
  - kind: thresholds
    thresholds: [0.941700041]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.235266, 0.703559637, 0.343733728]
)");
}

TEST_CASE("Generate MR-Sort model - fixed weights sum") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42, 2);

  std::ostringstream oss;
  model.dump(problem, oss);
  CHECK(oss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [0.377049327]
  - kind: thresholds
    thresholds: [0.790612161]
  - kind: thresholds
    thresholds: [0.941700041]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.366869569, 1.09711826, 0.536012173]
)");
}

TEST_CASE("Generate MR-Sort model - integer criteria") {
  Problem problem = generate_classification_problem(
    2, 2,
    535747649,
    false,
    {Criterion::PreferenceDirection::increasing, Criterion::PreferenceDirection::decreasing},
    {Criterion::ValueType::integer});
  Model model = generate_mrsort_classification_model(problem, 116273751);
  std::ostringstream oss;
  model.dump(problem, oss);
  CHECK(oss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [-1219]
  - kind: thresholds
    thresholds: [1634]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [0.620512545, 1.0907892]
)");
}

TEST_CASE("Generate MR-Sort model - enumerated criteria") {
  Problem problem = generate_classification_problem(
    2, 2,
    520326314,
    false,
    {Criterion::PreferenceDirection::increasing},  // @todo(Project management, later) Support an empty set of allowed preference directions
    {Criterion::ValueType::enumerated});
  Model model = generate_mrsort_classification_model(problem, 511376872);
  std::ostringstream oss;
  model.dump(problem, oss);
  CHECK(oss.str() == R"(kind: ncs-classification-model
format_version: 1
accepted_values:
  - kind: thresholds
    thresholds: [got]
  - kind: thresholds
    thresholds: [bir]
sufficient_coalitions:
  - kind: weights
    criterion_weights: [101.793587, 45.5191078]
)");
}

Alternatives generate_uniform_classified_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  std::mt19937& gen
) {
  CHRONE();

  const unsigned criteria_count = problem.get_criteria().size();

  std::vector<Alternative> alternatives;
  alternatives.reserve(alternatives_count);

  // We don't do anything to ensure homogeneous repartition among categories.
  // We just generate random profiles uniformly in [min, max] for each criterion
  std::map<unsigned, std::uniform_real_distribution<float>> real_values_distributions;
  std::map<unsigned, std::uniform_int_distribution<int>> int_values_distributions;
  std::map<unsigned, std::uniform_int_distribution<int>> enum_values_distributions;
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [&real_values_distributions, criterion_index](const Criterion::RealValues& values) {
        real_values_distributions[criterion_index] = std::uniform_real_distribution<float>(values.get_min_value(), values.get_max_value());
      },
      [&int_values_distributions, criterion_index](const Criterion::IntegerValues& values) {
        int_values_distributions[criterion_index] = std::uniform_int_distribution<int>(values.get_min_value(), values.get_max_value());
      },
      [&enum_values_distributions, criterion_index](const Criterion::EnumeratedValues& values) {
        enum_values_distributions[criterion_index] = std::uniform_int_distribution<int>(0, values.get_ordered_values().size() - 1);
      }
    );
  }

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    std::vector<Performance> profile;
    profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      profile.push_back(dispatch(
        problem.get_criteria()[criterion_index].get_values(),
        [&real_values_distributions, &gen, criterion_index](const Criterion::RealValues& values) {
          return Performance(Performance::Real(real_values_distributions[criterion_index](gen)));
        },
        [&int_values_distributions, &gen, criterion_index](const Criterion::IntegerValues& values) {
          return Performance(Performance::Integer(int_values_distributions[criterion_index](gen)));
        },
        [&enum_values_distributions, &gen, criterion_index](const Criterion::EnumeratedValues& values) {
          return Performance(Performance::Enumerated(values.get_ordered_values()[enum_values_distributions[criterion_index](gen)]));
        }
      ));
    }

    alternatives.push_back(Alternative{"", profile, std::nullopt});
  }

  Alternatives alts{problem, alternatives};
  classify_alternatives(problem, model, &alts);

  return alts;
}

unsigned min_category_size(const unsigned alternatives_count, const unsigned categories_count, const float max_imbalance) {
  assert(max_imbalance >= 0);
  assert(max_imbalance <= 1);

  return std::floor(alternatives_count * (1 - max_imbalance) / categories_count);
}

unsigned max_category_size(const unsigned alternatives_count, const unsigned categories_count, const float max_imbalance) {
  assert(max_imbalance >= 0);
  assert(max_imbalance <= 1);

  return std::ceil(alternatives_count * (1 + max_imbalance) / categories_count);
}

TEST_CASE("Balanced category sizes") {
  CHECK(min_category_size(100, 2, 0) == 50);
  CHECK(max_category_size(100, 2, 0) == 50);
  CHECK(min_category_size(100, 2, 1) == 0);
  CHECK(max_category_size(100, 2, 1) == 100);

  CHECK(min_category_size(100, 2, 0.2) == 40);
  CHECK(max_category_size(100, 2, 0.2) == 61);  // Should be 60, but floating point arithmetics...

  CHECK(min_category_size(100'000, 2, 0.2) == 40000);
  CHECK(max_category_size(100'000, 2, 0.2) == 60001);  // Should be 60000

  CHECK(min_category_size(150, 3, 0.2) == 40);
  CHECK(max_category_size(150, 3, 0.2) == 60);

  CHECK(min_category_size(100, 2, 0.3) == 35);
  CHECK(max_category_size(100, 2, 0.3) == 65);

  CHECK(min_category_size(99, 2, 0.2) == 39);
  CHECK(max_category_size(99, 2, 0.2) == 60);
}

Alternatives generate_balanced_classified_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  const float max_imbalance,
  std::mt19937& gen
) {
  CHRONE();

  assert(max_imbalance >= 0);
  assert(max_imbalance <= 1);

  const unsigned categories_count = problem.get_ordered_categories().size();

  // These parameters are somewhat arbitrary and not really critical,
  // but changing there values *does* change the generated set, because of the two-steps process below:
  // the same alternatives are generated in the same order, but they treated differently here.

  // How long to insist before accepting failure when we can't find any alternative for a given category
  const int max_iterations_with_no_effect_with_empty_category = 100;

  // How long to insist before accepting failure when we have found at least one alternative for each category
  const int max_iterations_with_no_effect_with_all_categories_populated = 1'000;

  // Size ratio to call 'uniform_learning_set'. Small values imply calling it more, and large values
  // imply discarding more alternatives
  const int multiplier = 10;

  const unsigned min_size = min_category_size(alternatives_count, categories_count, max_imbalance);
  const unsigned max_size = max_category_size(alternatives_count, categories_count, max_imbalance);

  std::vector<Alternative> alternatives;
  alternatives.reserve(alternatives_count);
  std::vector<unsigned> histogram(categories_count, 0);

  int max_iterations_with_no_effect = max_iterations_with_no_effect_with_empty_category;

  // Step 1: fill all categories to exactly the min size
  // (skip if min size is zero)
  int iterations_with_no_effect = 0;
  while (min_size > 0) {
    ++iterations_with_no_effect;

    Alternatives candidates = generate_uniform_classified_alternatives(problem, model, multiplier * alternatives_count, gen);

    for (const auto& candidate : candidates.get_alternatives()) {
      assert(candidate.get_category_index());
      const unsigned category_index = *candidate.get_category_index();
      if (histogram[category_index] < min_size) {
        alternatives.push_back(candidate);
        ++histogram[category_index];
        iterations_with_no_effect = 0;
      }
    }

    if (std::all_of(histogram.begin(), histogram.end(), [min_size](const auto size) { return size >= min_size; })) {
      // Success
      break;
    }

    if (std::all_of(histogram.begin(), histogram.end(), [](const auto size) { return size > 0; })) {
      max_iterations_with_no_effect = max_iterations_with_no_effect_with_all_categories_populated;
    }

    if (iterations_with_no_effect > max_iterations_with_no_effect) {
      throw BalancedAlternativesGenerationException(histogram);
    }
  }

  // Step 2: reach target size, keeping all categories below or at the max size
  iterations_with_no_effect = 0;
  while (true) {
    ++iterations_with_no_effect;

    Alternatives candidates = generate_uniform_classified_alternatives(problem, model, multiplier * alternatives_count, gen);

    for (const auto& candidate : candidates.get_alternatives()) {
      assert(candidate.get_category_index());
      const unsigned category_index = *candidate.get_category_index();
      if (histogram[category_index] < max_size) {
        alternatives.push_back(candidate);
        ++histogram[category_index];
        iterations_with_no_effect = 0;
      }

      if (alternatives.size() == alternatives_count) {
        assert(std::all_of(
          histogram.begin(), histogram.end(),
          [min_size, max_size](const auto size) { return size >= min_size && size <= max_size; }));
        return Alternatives(problem, alternatives);
      }
    }

    if (iterations_with_no_effect > max_iterations_with_no_effect) {
      throw BalancedAlternativesGenerationException(histogram);
    }
  }
}

Alternatives generate_classified_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  const unsigned random_seed,
  const std::optional<float> max_imbalance
) {
  CHRONE();

  std::mt19937 gen(random_seed);

  Alternatives alternatives = max_imbalance ?
    generate_balanced_classified_alternatives(problem, model, alternatives_count, *max_imbalance, gen) :
    generate_uniform_classified_alternatives(problem, model, alternatives_count, gen);

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    alternatives.get_writable_alternatives()[alternative_index].set_name("Alternative " + std::to_string(alternative_index + 1));
  }

  return alternatives;
}

TEST_CASE("Generate uniform alternative - integer criterion") {
  Problem problem = generate_classification_problem(
    1, 2,
    42,
    false,
    {Criterion::PreferenceDirection::increasing},
    {Criterion::ValueType::integer});
  Model model = generate_mrsort_classification_model(problem, 43);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 1, 44);

  CHECK(alternatives.get_alternatives()[0].get_profile()[0].get_integer().get_value() == 4537);
}

TEST_CASE("Generate uniform alternative - enumerated criterion") {
  Problem problem = generate_classification_problem(
    1, 2,
    42,
    false,
    {Criterion::PreferenceDirection::increasing},
    {Criterion::ValueType::enumerated});
  Model model = generate_mrsort_classification_model(problem, 43);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 1, 44);

  CHECK(alternatives.get_alternatives()[0].get_profile()[0].get_enumerated().get_value() == "put");
}

void check_histogram(const Problem& problem, const Model& model, const std::optional<float> max_imbalance, const unsigned a, const unsigned b) {
  const unsigned alternatives_count = 100;

  REQUIRE(problem.get_ordered_categories().size() == 2);
  REQUIRE(a + b == alternatives_count);

  Alternatives alternatives = generate_classified_alternatives(problem, model, alternatives_count, 42, max_imbalance);

  std::vector<unsigned> histogram(2, 0);
  for (const auto& alternative : alternatives.get_alternatives()) {
    ++histogram[*alternative.get_category_index()];
  }
  CHECK(histogram[0] == a);
  CHECK(histogram[1] == b);
}

TEST_CASE("Generate balanced classified alternatives") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42, 2);

  check_histogram(problem, model, std::nullopt, 80, 20);
  check_histogram(problem, model, 0.5, 63, 37);
  check_histogram(problem, model, 0.2, 56, 44);
  check_histogram(problem, model, 0.1, 54, 46);
  check_histogram(problem, model, 0.0, 50, 50);
}

TEST_CASE("Generate balanced classified alternatives - names are correct") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42, 2);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 100, 42, 0.);

  CHECK(alternatives.get_alternatives()[99].get_name() == "Alternative 100");
}

TEST_CASE("Generate balanced classified alternatives - many seeds") {
  // Assert that we can generate a balanced learning set for all generated models

  const unsigned alternatives_seed = 42;  // If we succeed with this arbitrary seed, we're confident we'll succeed with any seed
  const int max_model_seed = skip_long ? 10 : 100;

  // (dynamic OpenMP scheduling because iteration durations vary a lot)
  #pragma omp parallel for collapse(3) schedule(dynamic, 1)
  for (int criteria_count = 1; criteria_count < 7; ++criteria_count) {
    for (int categories_count = 2; categories_count < 7; ++categories_count) {
      for (int model_seed = 0; model_seed < max_model_seed; ++model_seed) {
        Problem problem = generate_classification_problem(criteria_count, categories_count, 42);

        CAPTURE(criteria_count);
        CAPTURE(categories_count);
        CAPTURE(model_seed);

        Model model = generate_mrsort_classification_model(problem, model_seed);

        // There *are* failures for larger numbers of criteria or categories,
        // but there is not much I can imagine doing to avoid that.
        bool expect_success = true;
        // Known failures: when the first (resp. last) profile and threshold are low (resp. high),
        // it's too difficult to find random alternatives in the first (resp. last) category.
        if (criteria_count == 3 && categories_count == 4 && model_seed == 24) expect_success = false;
        if (criteria_count == 5 && categories_count == 5 && model_seed == 45) expect_success = false;
        if (criteria_count == 5 && categories_count == 6 && model_seed == 8) expect_success = false;
        if (criteria_count == 6 && categories_count == 2 && model_seed == 87) expect_success = false;
        if (criteria_count == 6 && categories_count == 4 && model_seed == 21) expect_success = false;
        if (criteria_count == 6 && categories_count == 4 && model_seed == 43) expect_success = false;
        if (criteria_count == 6 && categories_count == 4 && model_seed == 52) expect_success = false;
        if (criteria_count == 6 && categories_count == 5 && model_seed == 8) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 11) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 14) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 26) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 29) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 42) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 54) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 76) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 78) expect_success = false;
        if (criteria_count == 6 && categories_count == 6 && model_seed == 96) expect_success = false;

        try {
          Alternatives alternatives = generate_classified_alternatives(problem, model, 100, alternatives_seed, 0);
          CHECK(expect_success);
        } catch (BalancedAlternativesGenerationException& e) {
          CHECK(!expect_success);
        }
      }
    }
  }
}

TEST_CASE("Random min/max") {
  Problem problem = generate_classification_problem(2, 2, 42, false);
  Model model = generate_mrsort_classification_model(problem, 42);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 1, 44);

  CHECK(problem.get_criteria()[0].get_real_values().get_min_value() == doctest::Approx(-25.092));
  CHECK(problem.get_criteria()[0].get_real_values().get_max_value() == doctest::Approx(59.3086));
  CHECK(model.get_accepted_values()[0].get_real_thresholds().get_thresholds()[0] == doctest::Approx(6.52194));
  CHECK(alternatives.get_alternatives()[0].get_profile()[0].get_real().get_value() == doctest::Approx(45.3692));

  CHECK(problem.get_criteria()[1].get_real_values().get_min_value() == doctest::Approx(-63.313));
  CHECK(problem.get_criteria()[1].get_real_values().get_max_value() == doctest::Approx(46.3988));
  CHECK(model.get_accepted_values()[1].get_real_thresholds().get_thresholds()[0] == doctest::Approx(24.0712));
  CHECK(alternatives.get_alternatives()[0].get_profile()[1].get_real().get_value() == doctest::Approx(-15.8581));
}

TEST_CASE("Decreasing criterion") {
  Problem problem = generate_classification_problem(
    1, 3,
    44,
    true,
    {Criterion::PreferenceDirection::increasing, Criterion::PreferenceDirection::decreasing}
  );
  Model model = generate_mrsort_classification_model(problem, 42);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 10, 44);

  CHECK(problem.get_criteria()[0].get_real_values().get_preference_direction() == Criterion::PreferenceDirection::decreasing);
  // Profiles are in decreasing order
  CHECK(model.get_accepted_values()[0].get_real_thresholds().get_thresholds()[0] == doctest::Approx(0.790612));
  CHECK(model.get_accepted_values()[0].get_real_thresholds().get_thresholds()[1] == doctest::Approx(0.377049));

  CHECK(alternatives.get_alternatives()[0].get_profile()[0].get_real().get_value() == doctest::Approx(0.834842));
  CHECK(*alternatives.get_alternatives()[0].get_category_index() == 0);

  CHECK(alternatives.get_alternatives()[1].get_profile()[0].get_real().get_value() == doctest::Approx(0.432542));
  CHECK(*alternatives.get_alternatives()[1].get_category_index() == 1);

  CHECK(alternatives.get_alternatives()[2].get_profile()[0].get_real().get_value() == doctest::Approx(0.104796));
  CHECK(*alternatives.get_alternatives()[2].get_category_index() == 2);
}

TEST_CASE("Exploratory test: 'std::shuffle' *can* keep something in place") {
  std::vector<unsigned> v(100);
  std::iota(v.begin(), v.end(), 0);

  CHECK(v[76] == 76);
  CHECK(v[77] == 77);
  CHECK(v[78] == 78);

  std::mt19937 gen(0);
  std::shuffle(v.begin(), v.end(), gen);

  CHECK(v[76] == 31);
  CHECK(v[77] == 77);  // Kept
  CHECK(v[78] == 71);
}

void misclassify_alternatives(const Problem& problem, Alternatives* alternatives, const unsigned count, const unsigned random_seed) {
  CHRONE();

  const unsigned categories_count = problem.get_ordered_categories().size();
  const unsigned alternatives_count = alternatives->get_alternatives().size();

  std::mt19937 gen(random_seed);

  std::vector<unsigned> alternative_indexes(alternatives_count);
  std::iota(alternative_indexes.begin(), alternative_indexes.end(), 0);
  std::shuffle(alternative_indexes.begin(), alternative_indexes.end(), gen);
  alternative_indexes.resize(count);

  for (const unsigned alternative_index : alternative_indexes) {
    auto& alternative = alternatives->get_writable_alternatives()[alternative_index];

    // Choose new index in [0, alternative.get_category_index() - 1] U [alternative.get_category_index() + 1, categories_count - 1]
    // => choose in [0, categories_count - 2] and increment if >= alternative.get_category_index()
    unsigned new_category_index = std::uniform_int_distribution<unsigned>(0, categories_count - 2)(gen);
    if (new_category_index >= *alternative.get_category_index()) {
      ++new_category_index;
    }

    alternative.set_category_index(new_category_index);
  }
}

TEST_CASE("Misclassify alternatives") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42, 2);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 100, 42, 0.2);

  misclassify_alternatives(problem, &alternatives, 10, 42);

  CHECK(classify_alternatives(problem, model, &alternatives).changed == 10);
}

} // namespace lincs
