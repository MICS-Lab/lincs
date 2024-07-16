// Copyright 2023-2024 Vincent Jacques

#include "accuracy-heuristic-on-gpu.hpp"

#include "../../../chrones.hpp"
#include "../../../randomness-utils.hpp"


namespace {

// Block size set to less than 1024 because we use more than 64 registers per thread and my GPU has only 64K registers
typedef GridFactory2D<256, 2> grid;

__device__
bool is_accepted(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned alternative_index
) {
  const unsigned alternative_rank = performance_ranks[criterion_index][alternative_index];
  const unsigned low_profile_rank = low_profile_ranks[model_index][boundary_index][criterion_index];
  if (single_peaked[criterion_index]) {
    const unsigned high_profile_rank = high_profile_ranks[model_index][boundary_index][high_profile_rank_indexes[criterion_index]];
    return low_profile_rank <= alternative_rank && alternative_rank <= high_profile_rank;
  } else {
    return low_profile_rank <= alternative_rank;
  }
}

__device__
unsigned get_assignment(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView2D<Device, const float> weights,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned alternative_index
) {
  const unsigned criteria_count = performance_ranks.s1();
  const unsigned categories_count = low_profile_ranks.s1() + 1;

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = categories_count - 1; category_index != 0; --category_index) {
    const unsigned boundary_index = category_index - 1;
    float accepted_weight = 0;
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      if (is_accepted(performance_ranks, single_peaked, low_profile_ranks, high_profile_rank_indexes, high_profile_ranks, model_index, boundary_index, criterion_index, alternative_index)) {
        accepted_weight += weights[model_index][criterion_index];
      }
    }
    if (accepted_weight >= 1) {
      return category_index;
    }
  }
  return 0;
}

__device__
void update_move_desirability_for_low_profile(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  const unsigned alternative_index,
  lincs::Desirability* desirability
) {
  const unsigned alternatives_count = performance_ranks.s0();
  const unsigned criteria_count = performance_ranks.s1();

  const unsigned current_rank = low_profile_ranks[model_index][boundary_index][criterion_index];
  const float weight = weights[model_index][criterion_index];

  const unsigned alternative_rank = performance_ranks[criterion_index][alternative_index];
  const unsigned learning_assignment = assignments[alternative_index];
  const unsigned model_assignment = get_assignment(
    performance_ranks,
    weights,
    single_peaked,
    low_profile_ranks,
    high_profile_rank_indexes,
    high_profile_ranks,
    model_index,
    alternative_index);

  float accepted_weight = 0;
  // There is a 'criterion_index' parameter above, *and* a local 'crit_index' just here
  for (unsigned crit_index = 0; crit_index != criteria_count; ++crit_index) {
    if (is_accepted(performance_ranks, single_peaked, low_profile_ranks, high_profile_rank_indexes, high_profile_ranks, model_index, boundary_index, crit_index, alternative_index)) {
      accepted_weight += weights[model_index][crit_index];
    }
  }

  if (destination_rank > current_rank) {
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight < 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight >= 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && accepted_weight - weight < 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment < boundary_index
      && model_assignment > boundary_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  } else {
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight >= 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight < 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && accepted_weight + weight >= 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && alternative_rank >= destination_rank
      && current_rank > alternative_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment > boundary_index + 1
      && model_assignment < boundary_index + 1
      && alternative_rank > destination_rank
      && current_rank >= alternative_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  }
}

// @todo(Performance, later) investigate how sharing preliminary computations done in all threads could improve perf
__global__
void compute_move_desirabilities_for_low_profile__kernel(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned actual_destinations_count,
  const ArrayView1D<Device, const unsigned> destination_ranks,
  ArrayView1D<Device, lincs::Desirability> desirabilities
) {
  const unsigned alt_index = grid::x();
  assert(alt_index < performance_ranks.s0() + grid::blockDim().x);
  const unsigned destination_index = grid::y();
  assert(destination_index < actual_destinations_count + grid::blockDim().y);

  // Map (embarrassingly parallel)
  if (alt_index < performance_ranks.s0() && destination_index < actual_destinations_count) {
    update_move_desirability_for_low_profile(
      performance_ranks,
      assignments,
      weights,
      single_peaked,
      low_profile_ranks,
      high_profile_rank_indexes,
      high_profile_ranks,
      model_index,
      boundary_index,
      criterion_index,
      destination_ranks[destination_index],
      alt_index,
      &desirabilities[destination_index]);
  }
}

__global__
void apply_best_move_for_low_profile__kernel(
  const ArrayView3D<Device, unsigned> low_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned actual_destinations_count,
  const ArrayView1D<Device, const unsigned> destination_ranks,
  const ArrayView1D<Device, const lincs::Desirability> desirabilities,
  const float desirability_threshold
) {
  // Single-key reduce
  // Could maybe be parallelized using divide and conquer? Or atomic compare-and-swap?
  unsigned best_destination_rank = destination_ranks[0];
  float best_desirability = desirabilities[0].value();
  for (unsigned destination_index = 1; destination_index < actual_destinations_count; ++destination_index) {
    const unsigned destination_rank = destination_ranks[destination_index];
    const float desirability = desirabilities[destination_index].value();

    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination_rank = destination_rank;
    }
  }

  if (best_desirability >= desirability_threshold) {
    low_profile_ranks[model_index][boundary_index][criterion_index] = best_destination_rank;
  }
}

__device__
void update_move_desirability_for_high_profile(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  const unsigned alternative_index,
  lincs::Desirability* desirability
) {
  const unsigned alternatives_count = performance_ranks.s0();
  const unsigned criteria_count = performance_ranks.s1();

  const unsigned current_rank = high_profile_ranks[model_index][boundary_index][high_profile_rank_indexes[criterion_index]];
  const float weight = weights[model_index][criterion_index];

  const unsigned alternative_rank = performance_ranks[criterion_index][alternative_index];
  const unsigned learning_assignment = assignments[alternative_index];
  const unsigned model_assignment = get_assignment(
    performance_ranks,
    weights,
    single_peaked,
    low_profile_ranks,
    high_profile_rank_indexes,
    high_profile_ranks,
    model_index,
    alternative_index);

  float accepted_weight = 0;
  // There is a 'criterion_index' parameter above, *and* a local 'crit_index' just here
  for (unsigned crit_index = 0; crit_index != criteria_count; ++crit_index) {
    if (is_accepted(performance_ranks, single_peaked, low_profile_ranks, high_profile_rank_indexes, high_profile_ranks, model_index, boundary_index, crit_index, alternative_index)) {
      accepted_weight += weights[model_index][crit_index];
    }
  }

  if (destination_rank < current_rank) {
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight < 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight >= 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index + 1
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
      && accepted_weight - weight < 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment < boundary_index
      && model_assignment > boundary_index
      && destination_rank < alternative_rank
      && alternative_rank <= current_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  } else {
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight >= 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == boundary_index + 1
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight < 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index
      && alternative_rank < destination_rank
      && current_rank < alternative_rank
      && accepted_weight + weight >= 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == boundary_index
      && model_assignment == boundary_index + 1
      && alternative_rank <= destination_rank
      && current_rank < alternative_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment > boundary_index + 1
      && model_assignment < boundary_index + 1
      && alternative_rank < destination_rank
      && current_rank <= alternative_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  }
}

// @todo(Performance, later) investigate how sharing preliminary computations done in all threads could improve perf
__global__
void compute_move_desirabilities_for_high_profile__kernel(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView1D<Device, const bool> single_peaked,
  const ArrayView3D<Device, const unsigned> low_profile_ranks,
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, const unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned actual_destinations_count,
  const ArrayView1D<Device, const unsigned> destination_ranks,
  ArrayView1D<Device, lincs::Desirability> desirabilities
) {
  const unsigned alt_index = grid::x();
  assert(alt_index < performance_ranks.s0() + grid::blockDim().x);
  const unsigned destination_index = grid::y();
  assert(destination_index < actual_destinations_count + grid::blockDim().y);

  // Map (embarrassingly parallel)
  if (alt_index < performance_ranks.s0() && destination_index < actual_destinations_count) {
    update_move_desirability_for_high_profile(
      performance_ranks,
      assignments,
      weights,
      single_peaked,
      low_profile_ranks,
      high_profile_rank_indexes,
      high_profile_ranks,
      model_index,
      boundary_index,
      criterion_index,
      destination_ranks[destination_index],
      alt_index,
      &desirabilities[destination_index]);
  }
}

__global__
void apply_best_move_for_high_profile__kernel(
  const ArrayView1D<Device, const unsigned> high_profile_rank_indexes,
  const ArrayView3D<Device, unsigned> high_profile_ranks,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned actual_destinations_count,
  const ArrayView1D<Device, const unsigned> destination_ranks,
  const ArrayView1D<Device, const lincs::Desirability> desirabilities,
  const float desirability_threshold
) {
  // Single-key reduce
  // Could maybe be parallelized using divide and conquer? Or atomic compare-and-swap?
  unsigned best_destination_rank = destination_ranks[0];
  float best_desirability = desirabilities[0].value();
  for (unsigned destination_index = 1; destination_index < actual_destinations_count; ++destination_index) {
    const unsigned destination_rank = destination_ranks[destination_index];
    const float desirability = desirabilities[destination_index].value();

    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination_rank = destination_rank;
    }
  }

  if (best_desirability >= desirability_threshold) {
    high_profile_ranks[model_index][boundary_index][high_profile_rank_indexes[criterion_index]] = best_destination_rank;
  }
}

}  // namespace

namespace lincs {

ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModelsBeingLearned::GpuModelsBeingLearned(const PreprocessedLearningSet& preprocessed_learning_set, const ModelsBeingLearned& host_models_being_learned) :
  performance_ranks(preprocessed_learning_set.performance_ranks.template clone_to<Device>()),
  assignments(preprocessed_learning_set.assignments.template clone_to<Device>()),
  single_peaked(preprocessed_learning_set.single_peaked.template clone_to<Device>()),
  weights(host_models_being_learned.models_count, preprocessed_learning_set.criteria_count, uninitialized),
  low_profile_ranks(host_models_being_learned.models_count, preprocessed_learning_set.boundaries_count, preprocessed_learning_set.criteria_count, uninitialized),
  high_profile_rank_indexes(host_models_being_learned.high_profile_rank_indexes.template clone_to<Device>()),
  high_profile_ranks(host_models_being_learned.models_count, preprocessed_learning_set.boundaries_count, host_models_being_learned.high_profile_ranks.s0(), uninitialized),
  desirabilities(host_models_being_learned.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::max_destinations_count, uninitialized),
  destination_ranks(host_models_being_learned.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::max_destinations_count, uninitialized)
{}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles(
  const unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  // Get optimized weights
  copy(host_models_being_learned.weights, ref(gpu_models_being_learned.weights));
  // Get (re-)initialized profiles
  copy(host_models_being_learned.low_profile_ranks, ref(gpu_models_being_learned.low_profile_ranks));
  copy(host_models_being_learned.high_profile_ranks, ref(gpu_models_being_learned.high_profile_ranks));

  #pragma omp parallel for
  for (int model_indexes_index = model_indexes_begin; model_indexes_index < model_indexes_end; ++model_indexes_index) {
    const unsigned model_index = host_models_being_learned.model_indexes[model_indexes_index];
    improve_model_profiles(model_index);
  }

  // Set improved profiles
  copy(gpu_models_being_learned.low_profile_ranks, ref(host_models_being_learned.low_profile_ranks));
  copy(gpu_models_being_learned.high_profile_ranks, ref(host_models_being_learned.high_profile_ranks));
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(preprocessed_learning_set.criteria_count, uninitialized);
  // Not worth parallelizing because models_being_learned.criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != preprocessed_learning_set.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned boundary_index = 0; boundary_index != preprocessed_learning_set.boundaries_count; ++boundary_index) {
    shuffle(host_models_being_learned.random_generators[model_index], ref(criterion_indexes));
    improve_boundary_profiles(model_index, boundary_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_boundary_profiles(
  const unsigned model_index,
  const unsigned boundary_index,
  const ArrayView1D<Host, const unsigned> criterion_indexes
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

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_low_profile_then_high_profile(
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
      host_models_being_learned.low_profile_ranks[model_index][boundary_index - 1][criterion_index],
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      host_models_being_learned.high_profile_ranks[model_index][boundary_index][host_models_being_learned.high_profile_rank_indexes[criterion_index]]:
      host_models_being_learned.low_profile_ranks[model_index][boundary_index + 1][criterion_index]
  );

  improve_high_profile(
    model_index,
    boundary_index,
    criterion_index,
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      host_models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] :
      host_models_being_learned.high_profile_ranks[model_index][boundary_index + 1][host_models_being_learned.high_profile_rank_indexes[criterion_index]],
    boundary_index == 0 ?
      preprocessed_learning_set.values_counts[criterion_index] - 1 :
      host_models_being_learned.high_profile_ranks[model_index][boundary_index - 1][host_models_being_learned.high_profile_rank_indexes[criterion_index]]
  );
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_low_profile_only(
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
      host_models_being_learned.low_profile_ranks[model_index][boundary_index - 1][criterion_index],
    boundary_index == preprocessed_learning_set.boundaries_count - 1 ?
      preprocessed_learning_set.values_counts[criterion_index] - 1 :
      host_models_being_learned.low_profile_ranks[model_index][boundary_index + 1][criterion_index]
  );
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_low_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned lowest_destination_rank,
  const unsigned highest_destination_rank
) {
  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(host_models_being_learned.low_profile_ranks[model_index][boundary_index][criterion_index] == lowest_destination_rank);
  } else {
    Array1D<Host, unsigned> host_destination_ranks(max_destinations_count, uninitialized);
    unsigned actual_destinations_count = 0;
    if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
      for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
        host_destination_ranks[destination_index] = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(host_models_being_learned.random_generators[model_index]);
      }
      actual_destinations_count = max_destinations_count;
    } else {
      for (unsigned destination_rank = lowest_destination_rank; destination_rank <= highest_destination_rank; ++destination_rank) {
        ++actual_destinations_count;
        const unsigned destination_index = destination_rank - lowest_destination_rank;
        host_destination_ranks[destination_index] = destination_rank;
      }
      assert(actual_destinations_count == highest_destination_rank - lowest_destination_rank + 1);
    }

    copy(host_destination_ranks, ref(gpu_models_being_learned.destination_ranks[model_index]));
    gpu_models_being_learned.desirabilities[model_index].fill_with_zeros();
    Grid grid = grid::make(preprocessed_learning_set.alternatives_count, actual_destinations_count);
    compute_move_desirabilities_for_low_profile__kernel<<<LOVE_CONFIG(grid)>>>(
      gpu_models_being_learned.performance_ranks,
      gpu_models_being_learned.assignments,
      gpu_models_being_learned.weights,
      gpu_models_being_learned.single_peaked,
      gpu_models_being_learned.low_profile_ranks,
      gpu_models_being_learned.high_profile_rank_indexes,
      gpu_models_being_learned.high_profile_ranks,
      model_index,
      boundary_index,
      criterion_index,
      actual_destinations_count,
      gpu_models_being_learned.destination_ranks[model_index],
      ref(gpu_models_being_learned.desirabilities[model_index]));
    check_last_cuda_error_sync_stream(cudaStreamDefault);

    apply_best_move_for_low_profile__kernel<<<1, 1>>>(
      ref(gpu_models_being_learned.low_profile_ranks),
      model_index,
      boundary_index,
      criterion_index,
      actual_destinations_count,
      gpu_models_being_learned.destination_ranks[model_index],
      gpu_models_being_learned.desirabilities[model_index],
      std::uniform_real_distribution<float>(0, 1)(host_models_being_learned.random_generators[model_index]));
    check_last_cuda_error_sync_stream(cudaStreamDefault);

    // @todo(Performance, later) Can we group this copying somehow?
    // Currently we copy just one float from device memory to host memory
    // (because just one float is potentialy modified by 'apply_best_move_for_low_profile__kernel',
    // and we need it back on the device for the next iteration)

    // Lov-e-CUDA doesn't provide a way to copy scalars, so we're back to the basics, using cudaMemcpy directly and doing pointer arithmetic.
    check_cuda_error(cudaMemcpy(
      host_models_being_learned.low_profile_ranks[model_index][boundary_index].data() + criterion_index,
      gpu_models_being_learned.low_profile_ranks[model_index][boundary_index].data() + criterion_index,
      1 * sizeof(unsigned),
      cudaMemcpyDeviceToHost));
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_high_profile(
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned lowest_destination_rank,
  const unsigned highest_destination_rank
) {
  assert(preprocessed_learning_set.single_peaked[criterion_index]);
  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(host_models_being_learned.high_profile_ranks[model_index][boundary_index][host_models_being_learned.high_profile_rank_indexes[criterion_index]] == lowest_destination_rank);
  } else {
    Array1D<Host, unsigned> host_destination_ranks(max_destinations_count, uninitialized);
    unsigned actual_destinations_count = 0;
    if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
      for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
        host_destination_ranks[destination_index] = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(host_models_being_learned.random_generators[model_index]);
      }
      actual_destinations_count = max_destinations_count;
    } else {
      for (unsigned destination_rank = lowest_destination_rank; destination_rank <= highest_destination_rank; ++destination_rank) {
        ++actual_destinations_count;
        const unsigned destination_index = destination_rank - lowest_destination_rank;
        host_destination_ranks[destination_index] = destination_rank;
      }
      assert(actual_destinations_count == highest_destination_rank - lowest_destination_rank + 1);
    }

    copy(host_destination_ranks, ref(gpu_models_being_learned.destination_ranks[model_index]));
    gpu_models_being_learned.desirabilities[model_index].fill_with_zeros();
    Grid grid = grid::make(preprocessed_learning_set.alternatives_count, actual_destinations_count);
    compute_move_desirabilities_for_high_profile__kernel<<<LOVE_CONFIG(grid)>>>(
      gpu_models_being_learned.performance_ranks,
      gpu_models_being_learned.assignments,
      gpu_models_being_learned.weights,
      gpu_models_being_learned.single_peaked,
      gpu_models_being_learned.low_profile_ranks,
      gpu_models_being_learned.high_profile_rank_indexes,
      gpu_models_being_learned.high_profile_ranks,
      model_index,
      boundary_index,
      criterion_index,
      actual_destinations_count,
      gpu_models_being_learned.destination_ranks[model_index],
      ref(gpu_models_being_learned.desirabilities[model_index]));
    check_last_cuda_error_sync_stream(cudaStreamDefault);

    apply_best_move_for_high_profile__kernel<<<1, 1>>>(
      gpu_models_being_learned.high_profile_rank_indexes,
      ref(gpu_models_being_learned.high_profile_ranks),
      model_index,
      boundary_index,
      criterion_index,
      actual_destinations_count,
      gpu_models_being_learned.destination_ranks[model_index],
      gpu_models_being_learned.desirabilities[model_index],
      std::uniform_real_distribution<float>(0, 1)(host_models_being_learned.random_generators[model_index]));
    check_last_cuda_error_sync_stream(cudaStreamDefault);

    assert(host_models_being_learned.high_profile_rank_indexes[criterion_index] < host_models_being_learned.high_profile_ranks.s0());
    check_cuda_error(cudaMemcpy(
      host_models_being_learned.high_profile_ranks[model_index][boundary_index].data() + host_models_being_learned.high_profile_rank_indexes[criterion_index],
      gpu_models_being_learned.high_profile_ranks[model_index][boundary_index].data() + host_models_being_learned.high_profile_rank_indexes[criterion_index],
      1 * sizeof(unsigned),
      cudaMemcpyDeviceToHost));
  }
}

}  // namespace lincs
