// Copyright 2023 Vincent Jacques

#include "probabilistic-maximal-discrimination-power-per-criterion.hpp"

#include "../../../chrones.hpp"
#include "../../../generation.hpp"  // Only for tests

#include "../../../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(LearningData& learning_data_) : learning_data(learning_data_) {
  CHRONE();

  rank_generators.reserve(learning_data.criteria_count);
  #ifndef NDEBUG  // Check pre-processing
  value_generators.reserve(learning_data.criteria_count);
  #endif  // Check pre-processing
  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    auto& rank_generator = rank_generators.emplace_back();
    rank_generator.reserve(learning_data.boundaries_count);
    #ifndef NDEBUG  // Check pre-processing
    auto& value_generator = value_generators.emplace_back();
    value_generator.reserve(learning_data.boundaries_count);
    #endif  // Check pre-processing

    // @todo(in branch topics/pre-process-learning-set, soon) Remove 'reversed': it was added only to keep an identical behavior, but it only impacts pseudo-random behavior.
    const bool reversed = learning_data.problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::decreasing;
    assert(reversed || learning_data.problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::growing);

    for (unsigned profile_index = 0; profile_index != learning_data.boundaries_count; ++profile_index) {
      #ifndef NDEBUG  // Check pre-processing
      auto [rank_probabilities, value_probabilities] = get_candidate_probabilities(criterion_index, profile_index);
      #else
      auto rank_probabilities = get_candidate_probabilities(criterion_index, profile_index);
      #endif  // Check pre-processing
      rank_generator.emplace_back(rank_probabilities, reversed);
      #ifndef NDEBUG  // Check pre-processing
      value_generator.emplace_back(value_probabilities, false);
      #endif  // Check pre-processing
    }
  }
}

#ifndef NDEBUG  // Check pre-processing
std::pair<std::map<unsigned, double>, std::map<float, double>>
#else
std::map<unsigned, double>
#endif  // Check pre-processing
InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities(
  unsigned criterion_index,
  unsigned profile_index
) {
  CHRONE();

  const Criterion& criterion = learning_data.problem.criteria[criterion_index];

  std::vector<std::pair<unsigned, float>> candidates_worse;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
  candidates_worse.reserve(learning_data.alternatives_count);
  std::vector<std::pair<unsigned, float>> candidates_better;
  candidates_better.reserve(learning_data.alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != learning_data.alternatives_count; ++alternative_index) {
    const unsigned rank = learning_data.performance_ranks[criterion_index][alternative_index];
    const float value = learning_data.sorted_values[criterion_index][rank];
    assert(value == learning_data.learning_alternatives[criterion_index][alternative_index]);
    const unsigned assignment = learning_data.assignments[alternative_index];
    if (assignment == profile_index) {
      candidates_worse.push_back({rank, value});
    } else if (assignment == profile_index + 1) {
      candidates_better.push_back({rank, value});
    }
  }

  if (candidates_better.empty() && candidates_worse.empty()) {
    // @todo(in branch topics/pre-process-learning-set) Always use rank = 0: this is a remnant of a previous weirdness where we always used min_value,
    // which corresponds to either rank 0 or last rank depending on the category correlation.
    const unsigned rank = criterion.category_correlation == Criterion::CategoryCorrelation::growing ? 0 : learning_data.values_counts[criterion_index] - 1;
    assert(learning_data.sorted_values[criterion_index][rank] == learning_data.problem.criteria[criterion_index].min_value);
    #ifndef NDEBUG  // Check pre-processing
    return {{{rank, 1.0}}, {{criterion.min_value, 1.0}}};
    #else
    return {{{rank, 1.0}}};
    #endif  // Check pre-processing
  } else {
    std::map<unsigned, double> rank_probabilities;
    #ifndef NDEBUG  // Check pre-processing
    std::map<float, double> value_probabilities;
    #endif  // Check pre-processing

    for (auto candidates : { candidates_worse, candidates_better }) {
      for (auto [candidate_rank, candidate_value] : candidates) {
        assert(learning_data.sorted_values[criterion_index][candidate_rank] == candidate_value);
        const bool already_evaluated = rank_probabilities.find(candidate_rank) != rank_probabilities.end();
        assert(already_evaluated == (value_probabilities.find(candidate_value) != value_probabilities.end()));
        if (already_evaluated) {
          // Candidate value has already been evaluated (because it appears several times)
          continue;
        }

        unsigned correctly_classified_count = 0;
        // @todo(Performance, later) Could we somehow sort 'candidates_worse' and 'candidates_better' and walk the values only once?
        // (Transforming this O(n²) loop in O(n*log n) + O(n))
        for (auto [rank, value] : candidates_worse) {
          const bool is_better = candidate_rank > rank;
          assert(is_better == criterion.strictly_better(candidate_value, value));
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        for (auto [rank, value] : candidates_better) {
          const bool is_better = rank >= candidate_rank;
          assert(is_better == criterion.better_or_equal(value, candidate_value));
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        const double probability = static_cast<double>(correctly_classified_count) / candidates.size();
        rank_probabilities[candidate_rank] = probability;
        #ifndef NDEBUG  // Check pre-processing
        value_probabilities[candidate_value] = probability;
        #endif  // Check pre-processing
      }
    }

    assert(!rank_probabilities.empty());
    #ifndef NDEBUG  // Check pre-processing
    assert(!value_probabilities.empty());
    assert(value_probabilities.size() == rank_probabilities.size());
    return {rank_probabilities, value_probabilities};
    #else
    return rank_probabilities;
    #endif  // Check pre-processing
  }
}

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const unsigned model_index = learning_data.model_indexes[model_indexes_begin];

    // Embarrassingly parallel
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      const Criterion& criterion = learning_data.problem.criteria[criterion_index];
      // Not parallel because of the profiles ordering constraint
      for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
        const unsigned profile_index = category_index - 1;
        #ifndef NDEBUG  // Check pre-processing
        std::mt19937 urbg(learning_data.urbgs[model_index]);  // Copy before using, to get the same random values
        #endif  // Check pre-processing
        unsigned rank = rank_generators[criterion_index][profile_index](learning_data.urbgs[model_index]);
        #ifndef NDEBUG  // Check pre-processing
        float value = value_generators[criterion_index][profile_index](urbg);
        assert(value == learning_data.sorted_values[criterion_index][rank]);
        #endif  // Check pre-processing

        // Enforce profiles ordering constraint
        if (criterion.category_correlation == Criterion::CategoryCorrelation::growing) {
          if (profile_index != learning_data.boundaries_count - 1) {
            rank = std::min(rank, learning_data.profile_ranks[criterion_index][profile_index + 1][model_index]);
            #ifndef NDEBUG  // Check pre-processing
            value = std::min(value, learning_data.profile_values[criterion_index][profile_index + 1][model_index]);
            assert(learning_data.sorted_values[criterion_index][rank] == value);
            #endif  // Check pre-processing
          }
        } else {
          assert(criterion.category_correlation == Criterion::CategoryCorrelation::decreasing);
          if (profile_index != learning_data.boundaries_count - 1) {
            rank = std::min(rank, learning_data.profile_ranks[criterion_index][profile_index + 1][model_index]);
            #ifndef NDEBUG  // Check pre-processing
            value = std::max(value, learning_data.profile_values[criterion_index][profile_index + 1][model_index]);
            assert(learning_data.sorted_values[criterion_index][rank] == value);
            #endif  // Check pre-processing
          }
        }

        assert(learning_data.sorted_values[criterion_index][rank] == value);
        learning_data.profile_ranks[criterion_index][profile_index][model_index] = rank;
        #ifndef NDEBUG  // Check pre-processing
        learning_data.profile_values[criterion_index][profile_index][model_index] = value;
        #endif  // Check pre-processing
      }
    }
  }
}

TEST_CASE("Initialize profiles - respect ordering") {
  Problem problem{
    {
      Criterion(
        "Criterion 1",
        Criterion::ValueType::real,
        Criterion::CategoryCorrelation::growing,
        0.0, 1.0
      ),
      Criterion(
        "Criterion 2",
        Criterion::ValueType::real,
        Criterion::CategoryCorrelation::decreasing,
        0.0, 1.0
      )
    },
    {
      Category("Category 1"),
      Category("Category 2"),
      Category("Category 3"),
    }
  };
  Model model = generate_mrsort_classification_model(problem, 42);
  auto learning_set = generate_classified_alternatives(problem, model, 1000, 42, 0.1);
  LearnMrsortByWeightsProfilesBreed::LearningData learning_data(problem, learning_set, 1, 42);
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion initializer(learning_data);

  for (unsigned iteration = 0; iteration != 10; ++iteration) {
    initializer.initialize_profiles(0, 1);
    // Both CHECKs fail at least once when the 'Enforce profiles ordering constraint' code is removed
    CHECK(learning_data.profile_values[0][0][0] <= learning_data.profile_values[0][1][0]);
    CHECK(learning_data.profile_ranks[0][0][0] <= learning_data.profile_ranks[0][1][0]);
    CHECK(learning_data.profile_values[1][0][0] >= learning_data.profile_values[1][1][0]);
    CHECK(learning_data.profile_ranks[1][0][0] <= learning_data.profile_ranks[1][1][0]);
  }
}


}  // namespace lincs
