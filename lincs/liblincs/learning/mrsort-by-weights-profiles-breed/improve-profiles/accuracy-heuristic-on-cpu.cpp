// Copyright 2023 Vincent Jacques

#include "accuracy-heuristic-on-cpu.hpp"

#include "../../../randomness-utils.hpp"


namespace lincs {

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles() {
  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != learning_data.models_count; ++model_index) {
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
  for (unsigned profile_index = 0; profile_index != learning_data.categories_count - 1; ++profile_index) {
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

  const int delta_profile = is_growing ? +1 : -1;
  const unsigned lowest_profile_index = is_growing ? 0 : learning_data.categories_count - 2;
  const unsigned highest_profile_index = is_growing ? learning_data.categories_count - 2 : 0;

  const float lowest_destination =
    profile_index == lowest_profile_index ?
      learning_data.problem.criteria[criterion_index].min_value :
      learning_data.profiles[criterion_index][profile_index - delta_profile][model_index];
  const float highest_destination =
    profile_index == highest_profile_index ?
      learning_data.problem.criteria[criterion_index].max_value :
      learning_data.profiles[criterion_index][profile_index + delta_profile][model_index];

  float best_destination = learning_data.profiles[criterion_index][profile_index][model_index];
  float best_desirability = Desirability().value();

  assert(lowest_destination <= highest_destination);
  if (lowest_destination == highest_destination) {
    assert(best_destination == lowest_destination);
    return;
  }

  // Not sure about this part: we're considering an arbitrary number of possible moves as described in
  // Mousseau's prez-mics-2018(8).pdf, but:
  //  - this is wasteful when there are fewer alternatives in the interval
  //  - this is not strictly consistent with, albeit much simpler than, Sobrie's thesis
  // @todo(Performance, later) Ask Vincent Mousseau about the following:
  // We could consider only a finite set of values for b_j described as follows:
  // - sort all the 'a_j's
  // - compute all midpoints between two successive 'a_j'
  // - add two extreme values (0 and 1, or above the greatest a_j and below the smallest a_j)
  // Then instead of taking a random values in [lowest_destination, highest_destination],
  // we'd take a random subset of the intersection of these midpoints with that interval.
  for (unsigned n = 0; n < 64; ++n) {
    // Map (embarrassingly parallel)
    float destination = highest_destination;
    // By specification, std::uniform_real_distribution should never return its highest value,
    // but "most existing implementations have a bug where they may occasionally" return it,
    // so we work around that bug by calling it again until it doesn't.
    // Ref: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    while (destination == highest_destination) {
      destination = std::uniform_real_distribution<float>(lowest_destination, highest_destination)(learning_data.urbgs[model_index]);
    }
    const float desirability = compute_move_desirability(
      model_index, profile_index, criterion_index, destination).value();
    // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  // @todo(Feature, soon) Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (std::uniform_real_distribution<float>(0, 1)(learning_data.urbgs[model_index]) <= best_desirability) {
    learning_data.profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

Desirability ImproveProfilesWithAccuracyHeuristicOnCpu::compute_move_desirability(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const float destination
) {
  Desirability d;

  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    update_move_desirability(
      model_index, profile_index, criterion_index, destination, alternative_index, &d);
  }

  return d;
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::update_move_desirability(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const float destination,
  const unsigned alternative_index,
  Desirability* desirability
) {
  const Criterion& criterion = learning_data.problem.criteria[criterion_index];

  const float current_position = learning_data.profiles[criterion_index][profile_index][model_index];
  const float weight = learning_data.weights[criterion_index][model_index];

  const float value = learning_data.learning_alternatives[criterion_index][alternative_index];
  const unsigned learning_assignment = learning_data.learning_assignments[alternative_index];
  const unsigned model_assignment = LearnMrsortByWeightsProfilesBreed::get_assignment(learning_data, model_index, alternative_index);

  // @todo(Project management, later) Factorize with get_assignment
  // (Same remark in accuracy-heuristic-on-gpu.cu)
  float weight_at_or_better_than_profile = 0;
  // There is a criterion parameter above, *and* a local criterion just here
  for (unsigned crit_index = 0; crit_index != learning_data.criteria_count; ++crit_index) {
    const Criterion& crit = learning_data.problem.criteria[crit_index];
    const float alternative_value = learning_data.learning_alternatives[crit_index][alternative_index];
    const float profile_value = learning_data.profiles[crit_index][profile_index][model_index];
    if (crit.better_or_equal(alternative_value, profile_value)) {
      weight_at_or_better_than_profile += learning_data.weights[crit_index][model_index];
    }
  }

  // These imbricated conditionals could be factorized, but this form has the benefit
  // of being a direct translation of the top of page 78 of Sobrie's thesis.
  // Correspondance:
  // - learning_assignment: bottom index of A*
  // - model_assignment: top index of A*
  // - profile_index: h
  // - destination: b_j +/- \delta
  // - current_position: b_j
  // - value: a_j
  // - weight_at_or_better_than_profile: \sigma
  // - weight: w_j
  // - 1: \lambda
  if (criterion.strictly_better(destination, current_position)) {
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && criterion.strictly_better(destination, value)
      && criterion.better_or_equal(value, current_position)
      && weight_at_or_better_than_profile - weight < 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && criterion.strictly_better(destination, value)
      && criterion.better_or_equal(value, current_position)
      && weight_at_or_better_than_profile - weight >= 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && criterion.strictly_better(destination, value)
      && criterion.better_or_equal(value, current_position)
      && weight_at_or_better_than_profile - weight < 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && criterion.strictly_better(destination, value)
      && criterion.better_or_equal(value, current_position)
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && criterion.strictly_better(destination, value)
      && criterion.better_or_equal(value, current_position)
    ) {
      ++desirability->t;
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && criterion.strictly_better(value, destination)
      && criterion.strictly_better(current_position, value)
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && criterion.strictly_better(value, destination)
      && criterion.strictly_better(current_position, value)
      && weight_at_or_better_than_profile + weight < 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && criterion.strictly_better(value, destination)
      && criterion.strictly_better(current_position, value)
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && criterion.better_or_equal(value, destination)
      && criterion.strictly_better(current_position, value)
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && criterion.strictly_better(value, destination)
      && criterion.better_or_equal(current_position, value)
    ) {
      ++desirability->t;
    }
  }
}

}  // namespace lincs
