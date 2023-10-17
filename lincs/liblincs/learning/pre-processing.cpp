// Copyright 2023 Vincent Jacques

#include "pre-processing.hpp"

#include <map>
#include <set>


namespace lincs {
PreProcessedLearningSet::PreProcessedLearningSet(
  const Problem& problem_,
  const Alternatives& learning_set_
) :
  problem(problem_),
  learning_set(learning_set_),
  criteria_count(problem.criteria.size()),
  categories_count(problem.categories.size()),
  boundaries_count(categories_count - 1),
  alternatives_count(learning_set.alternatives.size()),
  sorted_values(criteria_count, alternatives_count + 2, uninitialized),
  values_counts(criteria_count, uninitialized),
  performance_ranks(criteria_count, alternatives_count, uninitialized),
  assignments(alternatives_count, uninitialized)
{
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const bool is_growing = problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::growing;
    assert(is_growing || problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::decreasing);

    std::set<float> unique_values;

    unique_values.insert(problem.criteria[criterion_index].min_value);
    unique_values.insert(problem.criteria[criterion_index].max_value);
    for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives.size(); ++alternative_index) {
      unique_values.insert(learning_set.alternatives[alternative_index].profile[criterion_index]);
    }

    assert(unique_values.size() <= alternatives_count + 2);
    std::map<float, unsigned> value_ranks_for_criterion;
    for (float value : unique_values) {
      const unsigned value_rank = is_growing ? value_ranks_for_criterion.size() : unique_values.size() - value_ranks_for_criterion.size() - 1;
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

}  // namespace lincs
