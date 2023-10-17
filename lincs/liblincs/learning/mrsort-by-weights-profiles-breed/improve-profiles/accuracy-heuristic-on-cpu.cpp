// Copyright 2023 Vincent Jacques

#include "accuracy-heuristic-on-cpu.hpp"

#include "../../../chrones.hpp"
#include "../../../randomness-utils.hpp"


namespace lincs {

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles() {
  CHRONE();

  #pragma omp parallel for
  for (int model_index = 0; model_index < learning_data.models_count; ++model_index) {
    improve_model_profiles(model_index);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(learning_data.criteria_count, uninitialized);
  // Not worth parallelizing because criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != learning_data.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned profile_index = 0; profile_index != learning_data.boundaries_count; ++profile_index) {
    shuffle(learning_data.urbgs[model_index], ref(criterion_indexes));
    improve_model_profile(model_index, profile_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  ArrayView1D<Host, const unsigned> criterion_indexes
) {
  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned crit_idx_idx = 0; crit_idx_idx != learning_data.criteria_count; ++crit_idx_idx) {
    improve_model_profile(model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index
) {
  const Criterion& criterion = learning_data.problem.criteria[criterion_index];
  const bool is_growing = criterion.category_correlation == Criterion::CategoryCorrelation::growing;
  assert(is_growing || criterion.category_correlation == Criterion::CategoryCorrelation::decreasing);

  const float lowest_destination_rank =
    profile_index == 0 ?
      0 :
      learning_data.profile_ranks[criterion_index][profile_index - 1][model_index];
  const float highest_destination_rank =
    profile_index == learning_data.boundaries_count - 1 ?
      learning_data.values_counts[criterion_index] - 1 :
      learning_data.profile_ranks[criterion_index][profile_index + 1][model_index];

  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(learning_data.profile_ranks[criterion_index][profile_index][model_index] == lowest_destination_rank);
    return;
  }

  unsigned best_destination_rank = learning_data.profile_ranks[criterion_index][profile_index][model_index];
  #ifndef NDEBUG  // Check pre-processing
  float best_destination_value = learning_data.profile_values[criterion_index][profile_index][model_index];
  #endif  // Check pre-processing
  float best_desirability = Desirability().value();

  if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
    // We could try uniformly spread-out destinations instead of uniform random ones
    for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
      const unsigned destination_rank = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(learning_data.urbgs[model_index]);
      #ifndef NDEBUG  // Check pre-processing
      const float destination_value = learning_data.sorted_values[criterion_index][destination_rank];
      #endif  // Check pre-processing
      const float desirability = compute_move_desirability(
        model_index,
        profile_index,
        criterion_index,
        destination_rank
        #ifndef NDEBUG  // Check pre-processing
        ,
        destination_value
        #endif  // Check pre-processing
      ).value();
      // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
      if (desirability > best_desirability) {
        best_desirability = desirability;
        best_destination_rank = destination_rank;
        #ifndef NDEBUG  // Check pre-processing
        best_destination_value = destination_value;
        #endif  // Check pre-processing
      }
    }
  } else {
    for (unsigned destination_rank = lowest_destination_rank; destination_rank <= highest_destination_rank; ++destination_rank) {
      #ifndef NDEBUG  // Check pre-processing
      const float destination_value = learning_data.sorted_values[criterion_index][destination_rank];
      #endif  // Check pre-processing
      const float desirability = compute_move_desirability(
        model_index,
        profile_index,
        criterion_index,
        destination_rank
        #ifndef NDEBUG  // Check pre-processing
        ,
        destination_value
        #endif  // Check pre-processing
      ).value();
      // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
      if (desirability > best_desirability) {
        best_desirability = desirability;
        best_destination_rank = destination_rank;
        #ifndef NDEBUG  // Check pre-processing
        best_destination_value = destination_value;
        #endif  // Check pre-processing
      }
    }
  }

  assert(best_destination_value == learning_data.sorted_values[criterion_index][best_destination_rank]);

  // @todo(Project management, later) Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (std::uniform_real_distribution<float>(0, 1)(learning_data.urbgs[model_index]) <= best_desirability) {
    learning_data.profile_ranks[criterion_index][profile_index][model_index] = best_destination_rank;
    #ifndef NDEBUG  // Check pre-processing
    learning_data.profile_values[criterion_index][profile_index][model_index] = best_destination_value;
    #endif  // Check pre-processing
  }
}

Desirability ImproveProfilesWithAccuracyHeuristicOnCpu::compute_move_desirability(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const unsigned destination_rank
  #ifndef NDEBUG  // Check pre-processing
  ,
  const float destination_value
  #endif  // Check pre-processing
) {
  Desirability d;

  for (unsigned alternative_index = 0; alternative_index != learning_data.alternatives_count; ++alternative_index) {
    update_move_desirability(
      model_index,
      profile_index,
      criterion_index,
      destination_rank,
      #ifndef NDEBUG  // Check pre-processing
      destination_value,
      #endif  // Check pre-processing
      alternative_index,
      &d);
  }

  return d;
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::update_move_desirability(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  #ifndef NDEBUG  // Check pre-processing
  const float destination_value,
  #endif  // Check pre-processing
  const unsigned alternative_index,
  Desirability* desirability
) {
  assert(destination_value == learning_data.sorted_values[criterion_index][destination_rank]);

  const unsigned current_rank = learning_data.profile_ranks[criterion_index][profile_index][model_index];
  #ifndef NDEBUG  // Check pre-processing
  const float current_value = learning_data.profile_values[criterion_index][profile_index][model_index];
  assert(current_value == learning_data.sorted_values[criterion_index][current_rank]);
  #endif  // Check pre-processing
  const float weight = learning_data.weights[criterion_index][model_index];

  const unsigned alternative_rank = learning_data.performance_ranks[criterion_index][alternative_index];
  #ifndef NDEBUG  // Check pre-processing
  const float alternative_value = learning_data.sorted_values[criterion_index][alternative_rank];
  assert(alternative_value == learning_data.learning_alternatives[criterion_index][alternative_index]);
  #endif  // Check pre-processing
  const unsigned learning_assignment = learning_data.assignments[alternative_index];
  const unsigned model_assignment = LearnMrsortByWeightsProfilesBreed::get_assignment(learning_data, model_index, alternative_index);

  // @todo(Project management, later) Factorize with get_assignment
  // (Same remark in accuracy-heuristic-on-gpu.cu)
  float weight_at_or_better_than_profile = 0;
  // There is a criterion parameter above, *and* a local criterion just here
  for (unsigned crit_index = 0; crit_index != learning_data.criteria_count; ++crit_index) {
    const unsigned alternative_rank = learning_data.performance_ranks[crit_index][alternative_index];
    const unsigned profile_rank = learning_data.profile_ranks[crit_index][profile_index][model_index];
    const bool is_better = alternative_rank >= profile_rank;
    #ifndef NDEBUG  // Check pre-processing
    const Criterion& crit = learning_data.problem.criteria[crit_index];
    const float alternative_value = learning_data.sorted_values[crit_index][alternative_rank];
    assert(alternative_value == learning_data.learning_alternatives[crit_index][alternative_index]);
    const float profile_value = learning_data.sorted_values[crit_index][profile_rank];
    assert(profile_value == learning_data.profile_values[crit_index][profile_index][model_index]);
    assert(is_better == crit.better_or_equal(alternative_value, profile_value));
    #endif  // Check pre-processing
    if (is_better) {
      weight_at_or_better_than_profile += learning_data.weights[crit_index][model_index];
    }
  }

  // These imbricated conditionals could be factorized, but this form has the benefit
  // of being a direct translation of the top of page 78 of Sobrie's thesis.
  // Correspondance:
  // - learning_assignment: bottom index of A*
  // - model_assignment: top index of A*
  // - profile_index: h
  // - destination_value: b_j +/- \delta
  // - current_value: b_j
  // - alternative_value: a_j
  // - weight_at_or_better_than_profile: \sigma
  // - weight: w_j
  // - 1: \lambda
  #ifndef NDEBUG  // Check pre-processing
  const Criterion& criterion = learning_data.problem.criteria[criterion_index];
  assert((alternative_rank >= current_rank) == criterion.better_or_equal(alternative_value, current_value));
  assert((alternative_rank >= destination_rank) == criterion.better_or_equal(alternative_value, destination_value));
  assert((current_rank >= alternative_rank) == criterion.better_or_equal(current_value, alternative_value));
  assert((alternative_rank > destination_rank) == criterion.strictly_better(alternative_value, destination_value));
  assert((current_rank > alternative_rank) == criterion.strictly_better(current_value, alternative_value));
  assert((destination_rank > alternative_rank) == criterion.strictly_better(destination_value, alternative_value));
  assert((destination_rank > current_rank) == criterion.strictly_better(destination_value, current_value));
  #endif  // Check pre-processing

  if (destination_rank > current_rank) {
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight < 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight >= 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight < 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      ++desirability->t;
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight < 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && alternative_rank >= destination_rank
      && current_rank > alternative_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && alternative_rank > destination_rank
      && current_rank >= alternative_rank
    ) {
      ++desirability->t;
    }
  }
}

}  // namespace lincs
