// Copyright 2023-2024 Vincent Jacques

#include "probabilistic-maximal-discrimination-power-per-criterion.hpp"

#include "../../../chrones.hpp"
#include "../../../generation.hpp"  // Only for tests

#include "../../../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(LearningData& learning_data_) : learning_data(learning_data_) {
  CHRONE();

  low_rank_generators.reserve(learning_data.criteria_count);
  high_rank_generators.reserve(learning_data.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    auto& low_rank_generator = low_rank_generators.emplace_back();
    low_rank_generator.reserve(learning_data.boundaries_count);
    auto& high_rank_generator = high_rank_generators.emplace_back();
    high_rank_generator.reserve(learning_data.boundaries_count);

    for (unsigned boundary_index = 0; boundary_index != learning_data.boundaries_count; ++boundary_index) {
      low_rank_generator.emplace_back(get_candidate_probabilities_for_low_ranks(criterion_index, boundary_index));
      high_rank_generator.emplace_back(get_candidate_probabilities_for_high_ranks(criterion_index, boundary_index));
    }
  }
}

std::map<unsigned, double> InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities_for_low_ranks(
  unsigned criterion_index,
  unsigned profile_index
) {
  CHRONE();

  std::vector<unsigned> candidates_worse;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the focus for our optimization effort, we should measure.
  candidates_worse.reserve(learning_data.alternatives_count);
  std::vector<unsigned> candidates_better;
  candidates_better.reserve(learning_data.alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != learning_data.alternatives_count; ++alternative_index) {
    const unsigned rank = learning_data.performance_ranks[criterion_index][alternative_index];
    const unsigned assignment = learning_data.assignments[alternative_index];
    if (assignment == profile_index) {
      candidates_worse.push_back(rank);
    } else if (assignment == profile_index + 1) {
      candidates_better.push_back(rank);
    }
  }

  if (candidates_better.empty() && candidates_worse.empty()) {
    return {{{0, 1.0}}};
  } else {
    std::map<unsigned, double> rank_probabilities;

    for (auto candidates : { candidates_worse, candidates_better }) {
      for (auto candidate_rank : candidates) {
        const bool already_evaluated = rank_probabilities.find(candidate_rank) != rank_probabilities.end();
        if (already_evaluated) {
          // Candidate value has already been evaluated (because it appears several times)
          continue;
        }

        unsigned correctly_classified_count = 0;
        // @todo(Performance, later) Could we somehow sort 'candidates_worse' and 'candidates_better' and walk the values only once?
        // (Transforming this O(n²) loop in O(n*log n) + O(n))
        for (auto rank : candidates_worse) {
          const bool is_better = candidate_rank > rank;
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        for (auto rank : candidates_better) {
          const bool is_better = rank >= candidate_rank;
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        const double probability = static_cast<double>(correctly_classified_count) / candidates.size();
        rank_probabilities[candidate_rank] = probability;
      }
    }

    assert(!rank_probabilities.empty());
    return rank_probabilities;
  }
}

std::map<unsigned, double> InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities_for_high_ranks(
  unsigned criterion_index,
  unsigned profile_index
) {
  CHRONE();

  std::vector<unsigned> candidates_worse;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the focus for our optimization effort, we should measure.
  candidates_worse.reserve(learning_data.alternatives_count);
  std::vector<unsigned> candidates_better;
  candidates_better.reserve(learning_data.alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != learning_data.alternatives_count; ++alternative_index) {
    const unsigned rank = learning_data.performance_ranks[criterion_index][alternative_index];
    const unsigned assignment = learning_data.assignments[alternative_index];
    if (assignment == profile_index) {
      candidates_worse.push_back(rank);
    } else if (assignment == profile_index + 1) {
      candidates_better.push_back(rank);
    }
  }

  if (candidates_better.empty() && candidates_worse.empty()) {
    return {{{learning_data.values_counts[criterion_index] - 1, 1.0}}};
  } else {
    std::map<unsigned, double> rank_probabilities;

    for (auto candidates : { candidates_worse, candidates_better }) {
      for (auto candidate_rank : candidates) {
        const bool already_evaluated = rank_probabilities.find(candidate_rank) != rank_probabilities.end();
        if (already_evaluated) {
          // Candidate value has already been evaluated (because it appears several times)
          continue;
        }

        unsigned correctly_classified_count = 0;
        // @todo(Performance, later) Could we somehow sort 'candidates_worse' and 'candidates_better' and walk the values only once?
        // (Transforming this O(n²) loop in O(n*log n) + O(n))
        for (auto rank : candidates_worse) {
          const bool is_better = candidate_rank < rank;
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        for (auto rank : candidates_better) {
          const bool is_better = rank <= candidate_rank;
          if (is_better) {
            ++correctly_classified_count;
          }
        }
        const double probability = static_cast<double>(correctly_classified_count) / candidates.size();
        rank_probabilities[candidate_rank] = probability;
      }
    }

    assert(!rank_probabilities.empty());
    return rank_probabilities;
  }
}

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  const unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  // @todo(Performance, later) Parallelize these loops?
  // Embarrassingly parallel
  for (unsigned model_indexes_index = model_indexes_begin; model_indexes_index < model_indexes_end; ++model_indexes_index) {
    const unsigned model_index = learning_data.model_indexes[model_indexes_index];

    // Embarrassingly parallel
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      // Not parallel because of the profiles ordering constraint
      for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
        const unsigned boundary_index = category_index - 1;
        unsigned low_rank = low_rank_generators[criterion_index][boundary_index](learning_data.urbgs[model_index]);

        // Enforce profiles ordering constraint (1/2)
        if (boundary_index != learning_data.boundaries_count - 1) {
          low_rank = std::min(low_rank, learning_data.low_profile_ranks[model_index][boundary_index + 1][criterion_index]);
        }

        learning_data.low_profile_ranks[model_index][boundary_index][criterion_index] = low_rank;

        if (learning_data.single_peaked[criterion_index]) {
          unsigned high_rank = high_rank_generators[criterion_index][boundary_index](learning_data.urbgs[model_index]);

          // Enforce profiles ordering constraint (2/2)
          if (boundary_index == learning_data.boundaries_count - 1) {
            high_rank = std::max(high_rank, low_rank);
          } else {
            high_rank = std::max(high_rank, learning_data.high_profile_ranks[model_index][boundary_index + 1][criterion_index]);
          }

          learning_data.high_profile_ranks[model_index][boundary_index][criterion_index] = high_rank;
        }
      }
    }
  }
}

TEST_CASE("Initialize profiles - respect ordering") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::increasing, 0, 1)),
      Criterion("Criterion 2", Criterion::RealValues(Criterion::PreferenceDirection::decreasing, 0, 1)),
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
    // Both CHECKs fail at least once when the 'Enforce profiles ordering constraint (1/2)' code is removed
    CHECK(learning_data.low_profile_ranks[0][0][0] <= learning_data.low_profile_ranks[0][1][0]);
    CHECK(learning_data.low_profile_ranks[0][0][1] <= learning_data.low_profile_ranks[0][1][1]);
  }
}

TEST_CASE("Initialize profiles - respect ordering - single-peaked criteria") {
  Problem problem{
    {
      Criterion("Criterion 1", Criterion::RealValues(Criterion::PreferenceDirection::single_peaked, 0, 1)),
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
    // This CHECK fails at least once when the 'Enforce profiles ordering constraint (1/2)' code is removed
    CHECK(learning_data.low_profile_ranks[0][0][0] <= learning_data.low_profile_ranks[0][1][0]);
    // Both CHECKs fail at least once when the 'Enforce profiles ordering constraint (2/2)' code is removed
    CHECK(learning_data.low_profile_ranks[0][1][0] <= learning_data.high_profile_ranks[0][1][0]);
    CHECK(learning_data.high_profile_ranks[0][1][0] <= learning_data.high_profile_ranks[0][0][0]);
  }
}

}  // namespace lincs
