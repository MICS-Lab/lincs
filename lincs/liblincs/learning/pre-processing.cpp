// Copyright 2023-2024 Vincent Jacques

#include "pre-processing.hpp"

#include <cassert>
#include <map>
#include <set>


namespace lincs {

PreprocessedLearningSet::PreprocessedLearningSet(
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
  single_peaked(criteria_count, uninitialized),
  values_counts(criteria_count, uninitialized),
  performance_ranks(criteria_count, alternatives_count, uninitialized),
  assignments(alternatives_count, uninitialized)
{
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [this, &learning_set, criterion_index](const Criterion::RealValues& values) {
        const bool is_increasing = values.is_increasing() || values.is_single_peaked();
        assert(is_increasing || values.is_decreasing());

        single_peaked[criterion_index] = values.is_single_peaked();

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
        const bool is_increasing = values.is_increasing() || values.is_single_peaked();
        assert(is_increasing || values.is_decreasing());

        single_peaked[criterion_index] = values.is_single_peaked();

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

        single_peaked[criterion_index] = false;

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

Model PreprocessedLearningSet::post_process(const std::vector<PreprocessedBoundary>& boundaries) const {
  assert(boundaries.size() == boundaries_count);

  std::vector<AcceptedValues> accepted_values;
  accepted_values.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    accepted_values.push_back(dispatch(
      problem.get_criteria()[criterion_index].get_values(),
      [this, &boundaries, criterion_index](const Criterion::RealValues& values) {
        if (values.is_single_peaked()) {
          std::vector<std::optional<std::pair<float, float>>> intervals;
          intervals.reserve(boundaries_count);
          for (const auto& boundary: boundaries) {
            auto [low_rank, high_rank] = std::get<std::pair<unsigned, unsigned>>(boundary.profile_ranks[criterion_index]);
            if (low_rank < values_counts[criterion_index] && high_rank < values_counts[criterion_index]) {
              intervals.push_back(std::make_pair(real_sorted_values.at(criterion_index)[low_rank], real_sorted_values.at(criterion_index)[high_rank]));
            } else {
              intervals.emplace_back(std::nullopt);
            }
          }
          return AcceptedValues(AcceptedValues::RealIntervals(intervals));
        } else {
          std::vector<std::optional<float>> thresholds;
          thresholds.reserve(boundaries_count);
          for (const auto& boundary: boundaries) {
            const unsigned rank = std::get<unsigned>(boundary.profile_ranks[criterion_index]);
            if (rank < values_counts[criterion_index]) {
              thresholds.push_back(real_sorted_values.at(criterion_index)[rank]);
            } else {
              // Past-the-end rank => this criterion cannot help reach this category
              thresholds.push_back(std::nullopt);
            }
          }
          return AcceptedValues(AcceptedValues::RealThresholds(thresholds));
        }
      },
      [this, &boundaries, criterion_index](const Criterion::IntegerValues& values) {
        if (values.is_single_peaked()) {
          std::vector<std::optional<std::pair<int, int>>> intervals;
          intervals.reserve(boundaries_count);
          for (const auto& boundary: boundaries) {
            auto [low_rank, high_rank] = std::get<std::pair<unsigned, unsigned>>(boundary.profile_ranks[criterion_index]);
            // Handle past-the-end ranks
            low_rank = std::min(low_rank, values_counts[criterion_index] - 1);
            high_rank = std::min(high_rank, values_counts[criterion_index] - 1);
            intervals.push_back(std::make_pair(integer_sorted_values.at(criterion_index)[low_rank], integer_sorted_values.at(criterion_index)[high_rank]));
          }
          return AcceptedValues(AcceptedValues::IntegerIntervals(intervals));
        } else {
          std::vector<std::optional<int>> thresholds;
          thresholds.reserve(boundaries_count);
          for (const auto& boundary: boundaries) {
            unsigned rank = std::get<unsigned>(boundary.profile_ranks[criterion_index]);
            if (rank < values_counts[criterion_index]) {
              thresholds.push_back(integer_sorted_values.at(criterion_index)[rank]);
            } else {
              thresholds.push_back(std::nullopt);
            }
          }
          return AcceptedValues(AcceptedValues::IntegerThresholds(thresholds));
        }
      },
      [this, &boundaries, criterion_index](const Criterion::EnumeratedValues& values) {
        std::vector<std::optional<std::string>> thresholds;
        thresholds.reserve(boundaries_count);
        for (const auto& boundary: boundaries) {
          unsigned rank = std::get<unsigned>(boundary.profile_ranks[criterion_index]);
          if (rank < values_counts[criterion_index]) {
            thresholds.push_back(values.get_ordered_values()[rank]);
          } else {
            thresholds.push_back(std::nullopt);
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
