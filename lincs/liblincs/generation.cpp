// Copyright 2023 Vincent Jacques

#include "generation.hpp"

#include <algorithm>
#include <cassert>
#include <numeric>
#include <random>

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
  bool allow_decreasing_criteria
) {
  CHRONE();

  std::mt19937 gen(random_seed);
  std::uniform_real_distribution<float> min_max_distribution(-100, 100);
  std::uniform_int_distribution<unsigned> category_correlation_distribution(0, allow_decreasing_criteria ? 1 : 0);

  std::vector<Criterion> criteria;
  criteria.reserve(criteria_count);
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

    const Criterion::CategoryCorrelation correlation =
      category_correlation_distribution(gen) == 0 ?
      Criterion::CategoryCorrelation::growing :
      Criterion::CategoryCorrelation::decreasing;

    criteria.emplace_back(
      "Criterion " + std::to_string(criterion_index + 1),
      Criterion::ValueType::real,
      correlation,
      min_value, max_value
    );
  }

  std::vector<Category> categories;
  categories.reserve(categories_count);
  for (unsigned category_index = 0; category_index != categories_count; ++category_index) {
    categories.emplace_back(
      "Category " + std::to_string(category_index + 1)
    );
  }

  return Problem{criteria, categories};
}

Model generate_mrsort_classification_model(const Problem& problem, const unsigned random_seed, const std::optional<float> fixed_weights_sum) {
  CHRONE();

  const unsigned categories_count = problem.categories.size();
  const unsigned boundaries_count = categories_count - 1;
  const unsigned criteria_count = problem.criteria.size();

  std::mt19937 gen(random_seed);

  std::vector<std::vector<float>> profiles(boundaries_count, std::vector<float>(criteria_count));
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const auto& criterion = problem.criteria[criterion_index];
    // Profile can take any values. We arbitrarily generate them uniformly
    std::uniform_real_distribution<float> values_distribution(criterion.min_value + 0.01f, criterion.max_value - 0.01f);
    // (Values clamped strictly inside ']min, max[' to make it easier to generate balanced learning sets)
    // Profiles must be ordered on each criterion, so we generate a random column...
    std::vector<float> column(boundaries_count);
    std::generate(
      column.begin(), column.end(),
      [&values_distribution, &gen]() { return values_distribution(gen); });
    // ... sort it according to the criterion's correlation to categories...
    std::sort(column.begin(), column.end(), [&criterion](float left, float right) { return better_or_equal(criterion.category_correlation, right, left); });
    // ... and assign that column across all profiles.
    for (unsigned profile_index = 0; profile_index != boundaries_count; ++profile_index) {
      profiles[profile_index][criterion_index] = column[profile_index];
    }
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

  SufficientCoalitions coalitions{SufficientCoalitions::weights, denormalized_weights};

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned category_index = 0; category_index != boundaries_count; ++category_index) {
    boundaries.emplace_back(profiles[category_index], coalitions);
  }

  return Model(problem, boundaries);
}

TEST_CASE("Generate MR-Sort model - random weights sum") {
  Problem problem = generate_classification_problem(3, 2, 42);
  Model model = generate_mrsort_classification_model(problem, 42);

  // @todo(Project Management, later) Don't dump, test the structured model, to avoid breaking the test if the dump format changes
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

  // @todo(Project Management, later) Don't dump, test the structured model, to avoid breaking the test if the dump format changes
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

Alternatives generate_uniform_classified_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  std::mt19937& gen
) {
  CHRONE();

  const unsigned criteria_count = problem.criteria.size();

  std::vector<Alternative> alternatives;
  alternatives.reserve(alternatives_count);

  // We don't do anything to ensure homogeneous repartition amongst categories.
  // We just generate random profiles uniformly in [min, max] for each criterion
  std::vector<std::uniform_real_distribution<float>> values_distributions;
  values_distributions.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const auto& criterion = problem.criteria[criterion_index];
    assert(criterion.value_type == Criterion::ValueType::real);

    values_distributions.emplace_back(criterion.min_value, criterion.max_value);
  }

  for (unsigned alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      criteria_values[criterion_index] = values_distributions[criterion_index](gen);
    }

    alternatives.push_back(Alternative{"", criteria_values, std::nullopt});
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

  const unsigned categories_count = problem.categories.size();

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

    for (const auto& candidate : candidates.alternatives) {
      assert(candidate.category_index);
      const unsigned category_index = *candidate.category_index;
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

    for (const auto& candidate : candidates.alternatives) {
      assert(candidate.category_index);
      const unsigned category_index = *candidate.category_index;
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
    alternatives.alternatives[alternative_index].name = "Alternative " + std::to_string(alternative_index + 1);
  }

  return alternatives;
}

void check_histogram(const Problem& problem, const Model& model, const std::optional<float> max_imbalance, const unsigned a, const unsigned b) {
  REQUIRE(problem.categories.size() == 2);

  Alternatives alternatives = generate_classified_alternatives(problem, model, 100, 42, max_imbalance);

  std::vector<unsigned> histogram(2, 0);
  for (const auto& alternative : alternatives.alternatives) {
    ++histogram[*alternative.category_index];
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

  CHECK(alternatives.alternatives[99].name == "Alternative 100");
}

TEST_CASE("Generate balanced classified alternatives - many seeds") {
  // Assert that we can generate a balanced learning set for all generated models

  const unsigned alternatives_seed = 42;  // If we succeed with this arbitrary seed, we're confident we'll succeed with any seed
  const unsigned max_model_seed = skip_long ? 10 : 100;

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

  CHECK(problem.criteria[0].min_value == doctest::Approx(-25.092));
  CHECK(problem.criteria[0].max_value == doctest::Approx(59.3086));
  CHECK(model.boundaries[0].profile[0] == doctest::Approx(6.52194));
  CHECK(alternatives.alternatives[0].profile[0] == doctest::Approx(45.3692));

  CHECK(problem.criteria[1].min_value == doctest::Approx(-63.313));
  CHECK(problem.criteria[1].max_value == doctest::Approx(46.3988));
  CHECK(model.boundaries[0].profile[1] == doctest::Approx(24.0712));
  CHECK(alternatives.alternatives[0].profile[1] == doctest::Approx(-15.8581));
}

TEST_CASE("Decreasing criterion") {
  Problem problem = generate_classification_problem(1, 3, 44, true, true);
  Model model = generate_mrsort_classification_model(problem, 42);
  Alternatives alternatives = generate_classified_alternatives(problem, model, 10, 44);

  CHECK(problem.criteria[0].category_correlation == Criterion::CategoryCorrelation::decreasing);
  // Profiles are in decreasing order
  CHECK(model.boundaries[0].profile[0] == doctest::Approx(0.790612));
  CHECK(model.boundaries[1].profile[0] == doctest::Approx(0.377049));

  CHECK(alternatives.alternatives[0].profile[0] == doctest::Approx(0.834842));
  CHECK(*alternatives.alternatives[0].category_index == 0);

  CHECK(alternatives.alternatives[1].profile[0] == doctest::Approx(0.432542));
  CHECK(*alternatives.alternatives[1].category_index == 1);

  CHECK(alternatives.alternatives[2].profile[0] == doctest::Approx(0.104796));
  CHECK(*alternatives.alternatives[2].category_index == 2);
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

  const unsigned categories_count = problem.categories.size();
  const unsigned alternatives_count = alternatives->alternatives.size();

  std::mt19937 gen(random_seed);

  std::vector<unsigned> alternative_indexes(alternatives_count);
  std::iota(alternative_indexes.begin(), alternative_indexes.end(), 0);
  std::shuffle(alternative_indexes.begin(), alternative_indexes.end(), gen);
  alternative_indexes.resize(count);

  for (const unsigned alternative_index : alternative_indexes) {
    auto& alternative = alternatives->alternatives[alternative_index];

    // Choose new index in [0, alternative.category_index - 1] U [alternative.category_index + 1, categories_count - 1]
    // => choose in [0, categories_count - 2] and increment if >= alternative.category_index
    unsigned new_category_index = std::uniform_int_distribution<unsigned>(0, categories_count - 2)(gen);
    if (new_category_index >= *alternative.category_index) {
      ++new_category_index;
    }

    alternative.category_index = new_category_index;
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
