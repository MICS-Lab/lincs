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
  real_sorted_values(),
  integer_sorted_values(),
  values_counts(criteria_count, uninitialized),
  performance_ranks(criteria_count, alternatives_count, uninitialized),
  assignments(alternatives_count, uninitialized)
{
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    const bool is_increasing = criterion.get_preference_direction() == Criterion::PreferenceDirection::increasing;
    assert(is_increasing || criterion.get_preference_direction() == Criterion::PreferenceDirection::decreasing);

    switch (criterion.get_value_type()) {
      case Criterion::ValueType::real:
        {
          std::set<float> unique_values;

          unique_values.insert(criterion.get_real_min_value());
          unique_values.insert(criterion.get_real_max_value());
          for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
            unique_values.insert(learning_set.alternatives[alternative_index].profile[criterion_index].get_real_value());
          }
          const unsigned values_count = unique_values.size();
          values_counts[criterion_index] = values_count;

          real_sorted_values[criterion_index].resize(values_count);
          std::map<float, unsigned> value_ranks_for_criterion;
          for (const float value : unique_values) {
            const unsigned value_rank = is_increasing ? value_ranks_for_criterion.size() : values_count - value_ranks_for_criterion.size() - 1;
            real_sorted_values[criterion_index][value_rank] = value;
            value_ranks_for_criterion[value] = value_rank;
          }
          assert(value_ranks_for_criterion.size() == values_count);

          for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
            const float value = learning_set.alternatives[alternative_index].profile[criterion_index].get_real_value();
            const unsigned value_rank = value_ranks_for_criterion[value];
            performance_ranks[criterion_index][alternative_index] = value_rank;
          }
        }
        break;
      case Criterion::ValueType::integer:
        {
          std::set<int> unique_values;

          unique_values.insert(criterion.get_integer_min_value());
          unique_values.insert(criterion.get_integer_max_value());
          for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
            unique_values.insert(learning_set.alternatives[alternative_index].profile[criterion_index].get_integer_value());
          }
          const unsigned values_count = unique_values.size();
          values_counts[criterion_index] = values_count;

          integer_sorted_values[criterion_index].resize(values_count);
          std::map<int, unsigned> value_ranks_for_criterion;
          for (const int value : unique_values) {
            const unsigned value_rank = is_increasing ? value_ranks_for_criterion.size() : values_count - value_ranks_for_criterion.size() - 1;
            integer_sorted_values[criterion_index][value_rank] = value;
            value_ranks_for_criterion[value] = value_rank;
          }
          assert(value_ranks_for_criterion.size() == values_count);

          for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
            const int value = learning_set.alternatives[alternative_index].profile[criterion_index].get_integer_value();
            const unsigned value_rank = value_ranks_for_criterion[value];
            performance_ranks[criterion_index][alternative_index] = value_rank;
          }
        }
        break;
      case Criterion::ValueType::enumerated:
        {
          const std::vector<std::string> unique_values = criterion.get_ordered_values();
          const unsigned values_count = unique_values.size();
          values_counts[criterion_index] = values_count;

          std::map<std::string, unsigned> value_ranks_for_criterion;
          for (const std::string& value : unique_values) {
            const unsigned value_rank = value_ranks_for_criterion.size();
            value_ranks_for_criterion[value] = value_rank;
          }
          assert(value_ranks_for_criterion.size() == values_count);

          for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
            const std::string value = learning_set.alternatives[alternative_index].profile[criterion_index].get_enumerated_value();
            const unsigned value_rank = value_ranks_for_criterion[value];
            performance_ranks[criterion_index][alternative_index] = value_rank;
          }
        }
        break;
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
    const Criterion& criterion = problem.criteria[criterion_index];
    switch (criterion.get_value_type()) {
      case Criterion::ValueType::real:
        {
          std::vector<float> thresholds;
          thresholds.reserve(boundaries_count);
          for (const auto& boundary: model.boundaries) {
            const unsigned rank = boundary.profile_ranks[criterion_index];
            if (rank == 0) {
              thresholds.push_back(real_sorted_values.at(criterion_index)[rank]);
            } else if (rank == values_counts[criterion_index]) {
              // Past-the-end rank
              thresholds.push_back(real_sorted_values.at(criterion_index)[values_counts[criterion_index] - 1]);
            } else if (do_halves) {
              thresholds.push_back((real_sorted_values.at(criterion_index)[rank - 1] + real_sorted_values.at(criterion_index)[rank]) / 2);
            } else {
              thresholds.push_back(real_sorted_values.at(criterion_index)[rank]);
            }
          }
          accepted_values.push_back(AcceptedValues::make_real_thresholds(thresholds));
        }
        break;
      case Criterion::ValueType::integer:
        {
          std::vector<int> thresholds;
          thresholds.reserve(boundaries_count);
          for (const auto& boundary: model.boundaries) {
            const unsigned rank = boundary.profile_ranks[criterion_index];
            if (rank == 0) {
              thresholds.push_back(integer_sorted_values.at(criterion_index)[rank]);
            } else if (rank == values_counts[criterion_index]) {
              // Past-the-end rank
              thresholds.push_back(integer_sorted_values.at(criterion_index)[values_counts[criterion_index] - 1]);
            } else {
              thresholds.push_back(integer_sorted_values.at(criterion_index)[rank]);
            }
          }
          accepted_values.push_back(AcceptedValues::make_integer_thresholds(thresholds));
        }
        break;
      case Criterion::ValueType::enumerated:
        {
          const std::vector<std::string> ordered_values = criterion.get_ordered_values();
          std::vector<std::string> thresholds;
          thresholds.reserve(boundaries_count);
          for (const auto& boundary: model.boundaries) {
            const unsigned rank = boundary.profile_ranks[criterion_index];
            if (rank == 0) {
              thresholds.push_back(ordered_values[rank]);
            } else if (rank == values_counts[criterion_index]) {
              // Past-the-end rank
              thresholds.push_back(ordered_values[values_counts[criterion_index] - 1]);
            } else {
              thresholds.push_back(ordered_values[rank]);
            }
          }
          accepted_values.push_back(AcceptedValues::make_enumerated_thresholds(thresholds));
        }
        break;
    }
  }

  std::vector<SufficientCoalitions> sufficient_coalitions;
  sufficient_coalitions.reserve(boundaries_count);
  for (const auto& boundary: model.boundaries) {
    sufficient_coalitions.emplace_back(boundary.sufficient_coalitions);
  }

  return Model{problem, accepted_values, sufficient_coalitions};
}

}  // namespace lincs
