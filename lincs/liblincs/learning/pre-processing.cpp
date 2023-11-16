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
    const bool is_increasing = problem.criteria[criterion_index].get_preference_direction() == Criterion::PreferenceDirection::increasing;
    assert(is_increasing || problem.criteria[criterion_index].get_preference_direction() == Criterion::PreferenceDirection::decreasing);

    std::set<float> unique_values;

    unique_values.insert(problem.criteria[criterion_index].get_real_min_value());
    unique_values.insert(problem.criteria[criterion_index].get_real_max_value());
    for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives.size(); ++alternative_index) {
      unique_values.insert(learning_set.alternatives[alternative_index].profile[criterion_index]);
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

    for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives.size(); ++alternative_index) {
      const float value = learning_set.alternatives[alternative_index].profile[criterion_index];
      const unsigned value_rank = value_ranks_for_criterion[value];
      performance_ranks[criterion_index][alternative_index] = value_rank;
    }
  }

  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives.size(); ++alternative_index) {
    assignments[alternative_index] = *learning_set.alternatives[alternative_index].category_index;
  }
}

Model PreProcessedLearningSet::post_process(const PreProcessedModel& model, const bool do_halves) const {
  assert(model.boundaries.size() == boundaries_count);

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (const auto& boundary: model.boundaries) {
    assert(boundary.profile_ranks.size() == criteria_count);
    // @todo(Project management, later) Replace with:
    // profile.reserve(criteria_count)
    // for (rank : boundary.profile_ranks) {
    //   ... profile.emplace_back(...)
    std::vector<float> profile(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const unsigned rank = boundary.profile_ranks[criterion_index];
      if (rank == 0) {
        profile[criterion_index] = sorted_values[criterion_index][rank];
      } else if (rank == values_counts[criterion_index]) {
        // Past-the-end rank
        profile[criterion_index] = sorted_values[criterion_index][values_counts[criterion_index] - 1];
      } else if (do_halves) {
        profile[criterion_index] = (sorted_values[criterion_index][rank - 1] + sorted_values[criterion_index][rank]) / 2;
      } else {
        profile[criterion_index] = sorted_values[criterion_index][rank];
      }
    }
    boundaries.emplace_back(profile, boundary.sufficient_coalitions);
  }

  return Model{problem, boundaries};
}

}  // namespace lincs
