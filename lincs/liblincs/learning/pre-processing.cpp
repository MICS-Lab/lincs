// Copyright 2023 Vincent Jacques

#include "pre-processing.hpp"

#include <cassert>
#include <map>
#include <set>


namespace lincs {
PreProcessedLearningSet::PreProcessedLearningSet(
  const Problem& problem_,
  const Alternatives& learning_set
) :
  problem(problem_),
  criteria_count(problem.criteria.size()),
  categories_count(problem.ordered_categories.size()),
  boundaries_count(categories_count - 1),
  alternatives_count(learning_set.alternatives.size()),
  sorted_values(criteria_count, alternatives_count + 2, uninitialized),
  values_counts(criteria_count, uninitialized),
  performance_ranks(criteria_count, alternatives_count, uninitialized),
  assignments(alternatives_count, uninitialized)
{
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    const bool is_increasing = criterion.get_preference_direction() == Criterion::PreferenceDirection::increasing;
    assert(is_increasing || criterion.get_preference_direction() == Criterion::PreferenceDirection::decreasing);

    assert(criterion.is_real());
    std::set<float> unique_values;

    unique_values.insert(criterion.get_real_min_value());
    unique_values.insert(criterion.get_real_max_value());
    for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
      unique_values.insert(learning_set.alternatives[alternative_index].profile[criterion_index].get_real_value());
    }

    assert(unique_values.size() <= alternatives_count + 2);
    std::map<float, unsigned> value_ranks_for_criterion;
    for (float value : unique_values) {
      const unsigned value_rank = is_increasing ? value_ranks_for_criterion.size() : unique_values.size() - value_ranks_for_criterion.size() - 1;
      sorted_values[criterion_index][value_rank] = value;
      value_ranks_for_criterion[value] = value_rank;
    }
    assert(value_ranks_for_criterion.size() == unique_values.size());
    values_counts[criterion_index] = unique_values.size();

    for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
      const float value = learning_set.alternatives[alternative_index].profile[criterion_index].get_real_value();
      const unsigned value_rank = value_ranks_for_criterion[value];
      performance_ranks[criterion_index][alternative_index] = value_rank;
    }
  }

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    assignments[alternative_index] = *learning_set.alternatives[alternative_index].category_index;
  }
}

Model PreProcessedLearningSet::post_process(const PreProcessedModel& model, const bool do_halves) const {
  assert(model.boundaries.size() == boundaries_count);

  std::vector<AcceptedValues> accepted_values;
  accepted_values.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    assert(problem.criteria[criterion_index].is_real());
    std::vector<float> thresholds;
    thresholds.reserve(boundaries_count);
    for (const auto& boundary: model.boundaries) {
      const unsigned rank = boundary.profile_ranks[criterion_index];
      float threshold;
      if (rank == 0) {
        threshold = sorted_values[criterion_index][rank];
      } else if (rank == values_counts[criterion_index]) {
        // Past-the-end rank
        threshold = sorted_values[criterion_index][values_counts[criterion_index] - 1];
      } else if (do_halves) {
        threshold = (sorted_values[criterion_index][rank - 1] + sorted_values[criterion_index][rank]) / 2;
      } else {
        threshold = sorted_values[criterion_index][rank];
      }
      thresholds.push_back(threshold);
    }
    accepted_values.push_back(AcceptedValues::make_real_thresholds(thresholds));
  }

  std::vector<SufficientCoalitions> sufficient_coalitions;
  sufficient_coalitions.reserve(boundaries_count);
  for (const auto& boundary: model.boundaries) {
    sufficient_coalitions.emplace_back(boundary.sufficient_coalitions);
  }

  return Model{problem, accepted_values, sufficient_coalitions};
}

}  // namespace lincs
