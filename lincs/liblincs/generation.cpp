// Copyright 2023 Vincent Jacques

#include "generation.hpp"

#include <algorithm>
#include <cassert>
#include <random>

#include "classification.hpp"


namespace lincs {

Problem generate_problem(const unsigned criteria_count, const unsigned categories_count, const unsigned random_seed) {
  // There is nothing random yet. There will be when other value types and category correlations are added.

  std::vector<Problem::Criterion> criteria;
  criteria.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    criteria.emplace_back(
      "Criterion " + std::to_string(criterion_index + 1),
      Problem::Criterion::ValueType::real,
      Problem::Criterion::CategoryCorrelation::growing
    );
  }

  std::vector<Problem::Category> categories;
  categories.reserve(categories_count);
  for (unsigned category_index = 0; category_index != categories_count; ++category_index) {
    categories.emplace_back(
      "Category " + std::to_string(category_index + 1)
    );
  }

  return Problem{criteria, categories};
}

Model generate_mrsort_model(const Problem& problem, const unsigned random_seed, const std::optional<float> fixed_weights_sum) {
  const unsigned categories_count = problem.categories.size();
  const unsigned criteria_count = problem.criteria.size();

  std::mt19937 gen(random_seed);

  // Profile can take any values. We arbitrarily generate them uniformly
  std::uniform_real_distribution<float> values_distribution(0.01f, 0.99f);
  // (Values clamped strictly inside ']0, 1[' to make it easier to generate balanced learning sets)
  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (unsigned crit_index = 0; crit_index != criteria_count; ++crit_index) {
    // Profiles must be ordered on each criterion, so we generate a random column...
    std::vector<float> column(categories_count - 1);
    std::generate(
      column.begin(), column.end(),
      [&values_distribution, &gen]() { return values_distribution(gen); });
    // ... sort it...
    std::sort(column.begin(), column.end());
    // ... and assign that column across all profiles.
    for (unsigned profile_index = 0; profile_index != categories_count - 1; ++profile_index) {
      profiles[profile_index][crit_index] = column[profile_index];
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

  Model::SufficientCoalitions coalitions{
    Model::SufficientCoalitions::Kind::weights,
    denormalized_weights,
  };

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(categories_count - 1);
  for (unsigned category_index = 0; category_index != categories_count - 1; ++category_index) {
    boundaries.emplace_back(profiles[category_index], coalitions);
  }

  return Model(problem, boundaries);
}

Alternatives generate_uniform_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  std::mt19937& gen
) {
  const unsigned criteria_count = problem.criteria.size();

  std::vector<Alternative> alternatives;
  alternatives.reserve(alternatives_count);

  // We don't do anything to ensure homogeneous repartition amongst categories.
  // We just generate random profiles uniformly in [0, 1]
  std::uniform_real_distribution<float> values_distribution(0.0f, 1.0f);

  for (unsigned alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(criteria_count);
    std::generate(
      criteria_values.begin(), criteria_values.end(),
      [&values_distribution, &gen]() { return values_distribution(gen); });

    alternatives.push_back(Alternative{
      "Alternative " + std::to_string(alt_index + 1),
      criteria_values,
      std::nullopt,
    });
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

Alternatives generate_balanced_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  const float max_imbalance,
  std::mt19937& gen
) {
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
  std::map<std::string, unsigned> histogram;
  for (const auto& category : problem.categories) {
    histogram[category.name] = 0;
  }

  int max_iterations_with_no_effect = max_iterations_with_no_effect_with_empty_category;

  // Step 1: fill all categories to exactly the min size
  // (skip if min size is zero)
  int iterations_with_no_effect = 0;
  while (min_size > 0) {
    ++iterations_with_no_effect;

    Alternatives candidates = generate_uniform_alternatives(problem, model, multiplier * alternatives_count, gen);

    for (const auto& candidate : candidates.alternatives) {
      assert(candidate.category);
      const std::string& category = *candidate.category;
      if (histogram[category] < min_size) {
        alternatives.push_back(candidate);
        ++histogram[category];
        iterations_with_no_effect = 0;
      }
    }

    if (std::all_of(histogram.begin(), histogram.end(), [min_size](const auto it) { return it.second >= min_size; })) {
      // Success
      break;
    }

    if (std::all_of(histogram.begin(), histogram.end(), [](const auto it) { return it.second > 0; })) {
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

    Alternatives candidates = generate_uniform_alternatives(problem, model, multiplier * alternatives_count, gen);

    for (const auto& candidate : candidates.alternatives) {
      assert(candidate.category);
      const std::string& category = *candidate.category;
      if (histogram[category] < max_size) {
        alternatives.push_back(candidate);
        ++histogram[category];
        iterations_with_no_effect = 0;
      }

      if (alternatives.size() == alternatives_count) {
        assert(std::all_of(
          histogram.begin(), histogram.end(),
          [min_size, max_size](const auto it) { return it.second >= min_size && it.second <= max_size; }));
        return Alternatives(problem, alternatives);
      }
    }

    if (iterations_with_no_effect > max_iterations_with_no_effect) {
      throw BalancedAlternativesGenerationException(histogram);
    }
  }
}

Alternatives generate_alternatives(
  const Problem& problem,
  const Model& model,
  const unsigned alternatives_count,
  const unsigned random_seed,
  const std::optional<float> max_imbalance
) {
  std::mt19937 gen(random_seed);

  if (max_imbalance) {
    return generate_balanced_alternatives(problem, model, alternatives_count, *max_imbalance, gen);
  } else {
    return generate_uniform_alternatives(problem, model, alternatives_count, gen);
  }
}

} // namespace lincs
