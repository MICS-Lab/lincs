// Copyright 2023-2024 Vincent Jacques

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
  criteria_count(problem.get_criteria().size()),
  categories_count(problem.get_ordered_categories().size()),
  boundaries_count(categories_count - 1),
  alternatives_count(learning_set.get_alternatives().size()),
  real_sorted_values(),
  integer_sorted_values(),
  values_counts(criteria_count, uninitialized),
  performance_ranks(criteria_count, alternatives_count, uninitialized),
  assignments(alternatives_count, uninitialized)
{
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [this, &learning_set, criterion_index](const Criterion::RealValues& values) {
        const bool is_increasing = values.get_preference_direction() == Criterion::PreferenceDirection::increasing;
        assert(is_increasing || values.get_preference_direction() == Criterion::PreferenceDirection::decreasing);

        std::set<float> unique_values;

        unique_values.insert(values.get_min_value());
        unique_values.insert(values.get_max_value());
        for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
          unique_values.insert(learning_set.get_alternatives()[alternative_index].get_profile()[criterion_index].get_real().get_value());
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
        assert(value_ranks_for_criterion.size() == values_counts[criterion_index]);

        for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
          const float value = learning_set.get_alternatives()[alternative_index].get_profile()[criterion_index].get_real().get_value();
          const unsigned value_rank = value_ranks_for_criterion[value];
          performance_ranks[criterion_index][alternative_index] = value_rank;
        }
      },
      [this, &learning_set, criterion_index](const Criterion::IntegerValues& values) {
        const bool is_increasing = values.get_preference_direction() == Criterion::PreferenceDirection::increasing;
        assert(is_increasing || values.get_preference_direction() == Criterion::PreferenceDirection::decreasing);

        std::set<int> unique_values;

        unique_values.insert(values.get_min_value());
        unique_values.insert(values.get_max_value());
        for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
          unique_values.insert(learning_set.get_alternatives()[alternative_index].get_profile()[criterion_index].get_integer().get_value());
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
        assert(value_ranks_for_criterion.size() == values_counts[criterion_index]);

        for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
          const int value = learning_set.get_alternatives()[alternative_index].get_profile()[criterion_index].get_integer().get_value();
          const unsigned value_rank = value_ranks_for_criterion[value];
          performance_ranks[criterion_index][alternative_index] = value_rank;
        }
      },
      [this, &learning_set, criterion_index](const Criterion::EnumeratedValues& values) {
        values_counts[criterion_index] = values.get_ordered_values().size();

        std::map<std::string, unsigned> value_ranks_for_criterion;
        for (const std::string& value : values.get_ordered_values()) {
          const unsigned value_rank = value_ranks_for_criterion.size();
          value_ranks_for_criterion[value] = value_rank;
        }
        assert(value_ranks_for_criterion.size() == values_counts[criterion_index]);

        for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
          const std::string value = learning_set.get_alternatives()[alternative_index].get_profile()[criterion_index].get_enumerated().get_value();
          const unsigned value_rank = value_ranks_for_criterion[value];
          performance_ranks[criterion_index][alternative_index] = value_rank;
        }
      }
    );
  }

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    assignments[alternative_index] = *learning_set.get_alternatives()[alternative_index].get_category_index();
  }
}

Model PreProcessedLearningSet::post_process(const std::vector<PreProcessedBoundary>& boundaries) const {
  assert(boundaries.size() == boundaries_count);

  std::vector<AcceptedValues> accepted_values;
  accepted_values.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    accepted_values.push_back(dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [this, &boundaries, criterion_index](const Criterion::RealValues&) {
        std::vector<float> thresholds;
        thresholds.reserve(boundaries_count);
        for (const auto& boundary: boundaries) {
          const unsigned rank = boundary.profile_ranks[criterion_index];
          if (rank == 0) {
            thresholds.push_back(real_sorted_values.at(criterion_index)[rank]);
          } else if (rank == values_counts[criterion_index]) {
            // Past-the-end rank
            thresholds.push_back(real_sorted_values.at(criterion_index)[values_counts[criterion_index] - 1]);
          } else {
            thresholds.push_back(real_sorted_values.at(criterion_index)[rank]);
          }
        }
        return AcceptedValues(AcceptedValues::RealThresholds(thresholds));
      },
      [this, &boundaries, criterion_index](const Criterion::IntegerValues&) {
        std::vector<int> thresholds;
        thresholds.reserve(boundaries_count);
        for (const auto& boundary: boundaries) {
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
        return AcceptedValues(AcceptedValues::IntegerThresholds(thresholds));
      },
      [this, &boundaries, criterion_index](const Criterion::EnumeratedValues& values) {
        std::vector<std::string> thresholds;
        thresholds.reserve(boundaries_count);
        for (const auto& boundary: boundaries) {
          const unsigned rank = boundary.profile_ranks[criterion_index];
          if (rank == 0) {
            thresholds.push_back(values.get_ordered_values()[rank]);
          } else if (rank == values_counts[criterion_index]) {
            // Past-the-end rank
            thresholds.push_back(values.get_ordered_values()[values_counts[criterion_index] - 1]);
          } else {
            thresholds.push_back(values.get_ordered_values()[rank]);
          }
        }
        return AcceptedValues(AcceptedValues::EnumeratedThresholds(thresholds));
      }));
  }

  std::vector<SufficientCoalitions> sufficient_coalitions;
  sufficient_coalitions.reserve(boundaries_count);
  for (const auto& boundary: boundaries) {
    sufficient_coalitions.emplace_back(boundary.sufficient_coalitions);
  }

  return Model{problem, accepted_values, sufficient_coalitions};
}

}  // namespace lincs
