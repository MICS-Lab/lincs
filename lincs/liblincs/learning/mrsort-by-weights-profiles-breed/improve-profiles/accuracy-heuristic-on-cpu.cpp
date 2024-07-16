// Copyright 2023-2024 Vincent Jacques

#include "accuracy-heuristic-on-cpu.hpp"

#include "../../../chrones.hpp"
#include "../../../randomness-utils.hpp"


namespace lincs {

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles(
  const unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  const int model_indexes_end_ = model_indexes_end;

  #pragma omp parallel for
  for (int model_indexes_index = model_indexes_begin; model_indexes_index < model_indexes_end_; ++model_indexes_index) {
    const unsigned model_index = models_being_learned.model_indexes[model_indexes_index];
    improve_model_profiles(model_index);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(preprocessed_learning_set.criteria_count, uninitialized);
  // Not worth parallelizing because criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != preprocessed_learning_set.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned boundary_index = 0; boundary_index != preprocessed_learning_set.boundaries_count; ++boundary_index) {
    shuffle(models_being_learned.random_generators[model_index], ref(criterion_indexes));
    improve_boundary_profiles(model_index, boundary_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_boundary_profiles(
  const unsigned model_index,
  const unsigned boundary_index,
  ArrayView1D<Host, const unsigned> criterion_indexes
) {
  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned crit_idx_idx = 0; crit_idx_idx != preprocessed_learning_set.criteria_count; ++crit_idx_idx) {
    if (preprocessed_learning_set.single_peaked[criterion_indexes[crit_idx_idx]]) {
      improve_low_profile_then_high_profile(model_index, boundary_index, criterion_indexes[crit_idx_idx]);
    } else {
      improve_low_profile_only(model_index, boundary_index, criterion_indexes[crit_idx_idx]);
    }
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_low_profile_then_high_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index
) {
  assert(preprocessed_learning_set.single_peaked[criterion_index]);

  improve_low_profile(
    model_index,
    boundary_index,
    criterion_index,
    boundary_index == 0 ?
      0 :
      models_being_learned.low_profile_ranks[model_index][boundary_index - 1][criterion_index],
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      models_being_learned.high_profile_ranks[model_index][boundary_index][models_being_learned.high_profile_rank_indexes[criterion_index]]:
      models_being_learned.low_profile_ranks[model_index][boundary_index + 1][criterion_index]
  );

  improve_high_profile(
    model_index,
    boundary_index,
    criterion_index,
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] :
      models_being_learned.high_profile_ranks[model_index][boundary_index + 1][models_being_learned.high_profile_rank_indexes[criterion_index]],
    boundary_index == 0 ?
      preprocessed_learning_set.values_counts[criterion_index] - 1 :
      models_being_learned.high_profile_ranks[model_index][boundary_index - 1][models_being_learned.high_profile_rank_indexes[criterion_index]]
  );
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_low_profile_only(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index
) {
  improve_low_profile(
    model_index,
    boundary_index,
    criterion_index,
    boundary_index == 0 ?
      0 :
      models_being_learned.low_profile_ranks[model_index][boundary_index - 1][criterion_index],
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      preprocessed_learning_set.values_counts[criterion_index] - 1 :
      models_being_learned.low_profile_ranks[model_index][boundary_index + 1][criterion_index]
  );
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_low_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned lowest_destination_rank,
  const unsigned highest_destination_rank
) {
  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] == lowest_destination_rank);
  } else {
    unsigned best_destination_rank = models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index];
    float best_desirability = Desirability().value();

    if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
      // We could try uniformly spread-out destinations instead of uniform random ones
      for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
        const unsigned destination_rank = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(models_being_learned.random_generators[model_index]);
        const float desirability = compute_move_desirability_for_low_profile(
          model_index,
          boundary_index,
          criterion_index,
          destination_rank).value();
        // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
        if (desirability > best_desirability) {
          best_desirability = desirability;
          best_destination_rank = destination_rank;
        }
      }
    } else {
      for (unsigned destination_rank = lowest_destination_rank; destination_rank <= highest_destination_rank; ++destination_rank) {
        const float desirability = compute_move_desirability_for_low_profile(
          model_index,
          boundary_index,
          criterion_index,
          destination_rank).value();
        // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
        if (desirability > best_desirability) {
          best_desirability = desirability;
          best_destination_rank = destination_rank;
        }
      }
    }

    // @todo(Project management, later) Desirability can be as high as 2. The [0, 1] interval is a weird choice.
    if (std::uniform_real_distribution<float>(0, 1)(models_being_learned.random_generators[model_index]) <= best_desirability) {
      models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] = best_destination_rank;
    }
  }
}

Desirability ImproveProfilesWithAccuracyHeuristicOnCpu::compute_move_desirability_for_low_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank
) {
  Desirability d;

  for (unsigned alternative_index = 0; alternative_index != preprocessed_learning_set.alternatives_count; ++alternative_index) {
    update_move_desirability_for_low_profile(
      model_index,
      boundary_index,
      criterion_index,
      destination_rank,
      alternative_index,
      &d);
  }

  return d;
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::update_move_desirability_for_low_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  const unsigned alternative_index,
  Desirability* desirability
) {
  const unsigned current_rank = models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index];
  const float weight = models_being_learned.weights[model_index][criterion_index];

  const unsigned alternative_rank = preprocessed_learning_set.performance_ranks[criterion_index][alternative_index];
  const unsigned learning_assignment = preprocessed_learning_set.assignments[alternative_index];
  const unsigned model_assignment = LearnMrsortByWeightsProfilesBreed::get_assignment(preprocessed_learning_set, models_being_learned, model_index, alternative_index);

  // @todo(Project management, later) Factorize with get_assignment
  // (Same remark in accuracy-heuristic-on-gpu.cu)
  float accepted_weight = 0;
  // There is a 'criterion_index' parameter above, *and* a local 'crit_index' just here
  for (unsigned crit_index = 0; crit_index != preprocessed_learning_set.criteria_count; ++crit_index) {
    if (LearnMrsortByWeightsProfilesBreed::is_accepted(preprocessed_learning_set, models_being_learned, model_index, boundary_index, crit_index, alternative_index)) {
      accepted_weight += models_being_learned.weights[model_index][crit_index];
    }
  }

  // These imbricated conditionals could be factorized, but this form has the benefit
  // of being a direct translation of the top of page 78 of Sobrie's thesis.
  // Correspondance:
  // - learning_assignment: bottom index of A*
  // - model_assignment: top index of A*
  // - boundary_index: h
  // - destination_value: b_j +/- \delta
  // - current_value: b_j
  // - alternative_value: a_j
  // - accepted_weight: \sigma
  // - weight: w_j
  // - 1: \lambda
  if (destination_rank > current_rank) {
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight < 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight >= 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight < 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment < boundary_index
      && model_assignment > boundary_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      ++desirability->t;
    }
  } else {
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight >= 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight < 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight >= 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && alternative_rank >= destination_rank
      && current_rank > alternative_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment > boundary_index + 1
      && model_assignment < boundary_index + 1
      && alternative_rank > destination_rank
      && current_rank >= alternative_rank
    ) {
      ++desirability->t;
    }
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_high_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned lowest_destination_rank,
  const unsigned highest_destination_rank
) {
  assert(preprocessed_learning_set.single_peaked[criterion_index]);
  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(models_being_learned.high_profile_ranks[model_index][boundary_index][models_being_learned.high_profile_rank_indexes[criterion_index]] == lowest_destination_rank);
  } else {
    unsigned best_destination_rank = models_being_learned.high_profile_ranks[model_index][boundary_index][models_being_learned.high_profile_rank_indexes[criterion_index]];
    float best_desirability = Desirability().value();

    if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
      for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
        const unsigned destination_rank = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(models_being_learned.random_generators[model_index]);
        const float desirability = compute_move_desirability_for_high_profile(
          model_index,
          boundary_index,
          criterion_index,
          destination_rank).value();
        if (desirability > best_desirability) {
          best_desirability = desirability;
          best_destination_rank = destination_rank;
        }
      }
    } else {
      for (unsigned destination_rank = lowest_destination_rank; destination_rank <= highest_destination_rank; ++destination_rank) {
        const float desirability = compute_move_desirability_for_high_profile(
          model_index,
          boundary_index,
          criterion_index,
          destination_rank).value();
        if (desirability > best_desirability) {
          best_desirability = desirability;
          best_destination_rank = destination_rank;
        }
      }
    }

    if (std::uniform_real_distribution<float>(0, 1)(models_being_learned.random_generators[model_index]) <= best_desirability) {
      models_being_learned.high_profile_ranks[model_index][boundary_index][models_being_learned.high_profile_rank_indexes[criterion_index]] = best_destination_rank;
    }
  }
}

Desirability ImproveProfilesWithAccuracyHeuristicOnCpu::compute_move_desirability_for_high_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank
) {
  Desirability d;

  for (unsigned alternative_index = 0; alternative_index != preprocessed_learning_set.alternatives_count; ++alternative_index) {
    update_move_desirability_for_high_profile(
      model_index,
      boundary_index,
      criterion_index,
      destination_rank,
      alternative_index,
      &d);
  }

  return d;
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::update_move_desirability_for_high_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  const unsigned alternative_index,
  Desirability* desirability
) {
  assert(preprocessed_learning_set.single_peaked[criterion_index]);
  const unsigned current_rank = models_being_learned.high_profile_ranks[model_index][boundary_index][models_being_learned.high_profile_rank_indexes[criterion_index]];
  const float weight = models_being_learned.weights[model_index][criterion_index];

  const unsigned alternative_rank = preprocessed_learning_set.performance_ranks[criterion_index][alternative_index];
  const unsigned learning_assignment = preprocessed_learning_set.assignments[alternative_index];
  const unsigned model_assignment = LearnMrsortByWeightsProfilesBreed::get_assignment(preprocessed_learning_set, models_being_learned, model_index, alternative_index);

  float accepted_weight = 0;
  // There is a 'criterion_index' parameter above, *and* a local 'crit_index' just here
  for (unsigned crit_index = 0; crit_index != preprocessed_learning_set.criteria_count; ++crit_index) {
    if (LearnMrsortByWeightsProfilesBreed::is_accepted(preprocessed_learning_set, models_being_learned, model_index, boundary_index, crit_index, alternative_index)) {
      accepted_weight += models_being_learned.weights[model_index][crit_index];
    }
  }

  // Similar to 'update_move_desirability_for_low_profile' but inequalities involving 'destination_rank' and/or 'current_rank' are reversed
  if (destination_rank < current_rank) {
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight < 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight >= 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight < 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment < boundary_index
      && model_assignment > boundary_index
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
    ) {
      ++desirability->t;
    }
  } else {
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight >= 1
    ) {
      ++desirability->v;
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight < 1
    ) {
      ++desirability->w;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight >= 1
    ) {
      ++desirability->q;
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && alternative_rank <= destination_rank
      && current_rank < alternative_rank
    ) {
      ++desirability->r;
    }
    if (
      learning_assignment > boundary_index + 1
      && model_assignment < boundary_index + 1
      && alternative_rank < destination_rank
      && current_rank <= alternative_rank
    ) {
      ++desirability->t;
    }
  }
}

}  // namespace lincs
