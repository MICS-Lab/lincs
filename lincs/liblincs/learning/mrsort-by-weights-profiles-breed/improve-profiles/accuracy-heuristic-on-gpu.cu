// Copyright 2023-2024 Vincent Jacques

#include "accuracy-heuristic-on-gpu.hpp"

#include "../../../chrones.hpp"
#include "../../../randomness-utils.hpp"


namespace {

// Block size set to less than 1024 because we use more than 64 registers per thread and my GPU has only 64K registers
typedef GridFactory2D<256, 2> grid;

__device__
unsigned get_assignment(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const unsigned> profile_ranks,
  const unsigned model_index,
  const unsigned alternative_index
) {
  const unsigned criteria_count = performance_ranks.s1();
  const unsigned categories_count = profile_ranks.s1() + 1;

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_better_than_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const unsigned alternative_rank = performance_ranks[criterion_index][alternative_index];
      const unsigned profile_rank = profile_ranks[model_index][profile_index][criterion_index];
      if (alternative_rank >= profile_rank) {
        weight_at_or_better_than_profile += weights[model_index][criterion_index];
      }
    }
    if (weight_at_or_better_than_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

__device__
void update_move_desirability(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const unsigned> profile_ranks,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const unsigned destination_rank,
  const unsigned alternative_index,
  lincs::Desirability* desirability
) {
  const unsigned alternatives_count = performance_ranks.s0();
  const unsigned criteria_count = performance_ranks.s1();

  const unsigned current_rank = profile_ranks[model_index][profile_index][criterion_index];
  const float weight = weights[model_index][criterion_index];

  const unsigned alternative_rank = performance_ranks[criterion_index][alternative_index];
  const unsigned learning_assignment = assignments[alternative_index];
  const unsigned model_assignment = get_assignment(
    performance_ranks,
    weights,
    profile_ranks,
    model_index,
    alternative_index);

  float weight_at_or_better_than_profile = 0;
  // There is a criterion parameter above, *and* a local criterion just here
  for (unsigned crit_index = 0; crit_index != criteria_count; ++crit_index) {
    const unsigned alternative_rank = performance_ranks[crit_index][alternative_index];
    const unsigned profile_rank = profile_ranks[model_index][profile_index][crit_index];
    if (alternative_rank >= profile_rank) {
      weight_at_or_better_than_profile += weights[model_index][crit_index];
    }
  }

  if (destination_rank > current_rank) {
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight < 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight >= 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
      && weight_at_or_better_than_profile - weight < 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && destination_rank > alternative_rank
      && alternative_rank >= current_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      atomicInc(&desirability->v, alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight < 1
    ) {
      atomicInc(&desirability->w, alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && alternative_rank > destination_rank
      && current_rank > alternative_rank
      && weight_at_or_better_than_profile + weight >= 1
    ) {
      atomicInc(&desirability->q, alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && alternative_rank >= destination_rank
      && current_rank > alternative_rank
    ) {
      atomicInc(&desirability->r, alternatives_count);
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && alternative_rank > destination_rank
      && current_rank >= alternative_rank
    ) {
      atomicInc(&desirability->t, alternatives_count);
    }
  }
}

// @todo(Performance, later) investigate how sharing preliminary computations done in all threads could improve perf
__global__
void compute_move_desirabilities__kernel(
  const ArrayView2D<Device, const unsigned> performance_ranks,
  const ArrayView1D<Device, const unsigned> assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const unsigned> profile_ranks,
  const unsigned model_index,
  const unsigned profile_index,
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
    update_move_desirability(
      performance_ranks,
      assignments,
      weights,
      profile_ranks,
      model_index,
      profile_index,
      criterion_index,
      destination_ranks[destination_index],
      alt_index,
      &desirabilities[destination_index]);
  }
}

__global__
void apply_best_move__kernel(
  const ArrayView3D<Device, unsigned> profile_ranks,
  const unsigned model_index,
  const unsigned profile_index,
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
    profile_ranks[model_index][profile_index][criterion_index] = best_destination_rank;
  }
}

}  // namespace

namespace lincs {

ImproveProfilesWithAccuracyHeuristicOnGpu::GpuLearningData::GpuLearningData(const LearningData& host_learning_data) :
  performance_ranks(host_learning_data.performance_ranks.template clone_to<Device>()),
  assignments(host_learning_data.assignments.template clone_to<Device>()),
  weights(host_learning_data.models_count, host_learning_data.criteria_count, uninitialized),
  profile_ranks(host_learning_data.models_count, host_learning_data.boundaries_count, host_learning_data.criteria_count, uninitialized),
  desirabilities(host_learning_data.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::max_destinations_count, uninitialized),
  destination_ranks(host_learning_data.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::max_destinations_count, uninitialized)
{}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles() {
  CHRONE();

  // Get optimized weights
  copy(host_learning_data.weights, ref(gpu_learning_data.weights));
  // Get (re-)initialized profiles
  copy(host_learning_data.profile_ranks, ref(gpu_learning_data.profile_ranks));

  #pragma omp parallel for
  for (int model_index = 0; model_index < host_learning_data.models_count; ++model_index) {
    improve_model_profiles(model_index);
  }

  // Set improved profiles
  copy(gpu_learning_data.profile_ranks, ref(host_learning_data.profile_ranks));
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(host_learning_data.criteria_count, uninitialized);
  // Not worth parallelizing because learning_data.criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != host_learning_data.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned profile_index = 0; profile_index != host_learning_data.boundaries_count; ++profile_index) {
    shuffle(host_learning_data.urbgs[model_index], ref(criterion_indexes));
    improve_model_profile(model_index, profile_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  const ArrayView1D<Host, const unsigned> criterion_indexes
) {
  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned crit_idx_idx = 0; crit_idx_idx != host_learning_data.criteria_count; ++crit_idx_idx) {
    improve_model_profile(model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index
) {
  auto& learning_data = host_learning_data;

  const float lowest_destination_rank =
    profile_index == 0 ?
      0 :
      learning_data.profile_ranks[model_index][profile_index - 1][criterion_index];
  const float highest_destination_rank =
    profile_index == learning_data.boundaries_count - 1 ?
      learning_data.values_counts[criterion_index] - 1 :
      learning_data.profile_ranks[model_index][profile_index + 1][criterion_index];

  assert(lowest_destination_rank <= highest_destination_rank);
  if (lowest_destination_rank == highest_destination_rank) {
    assert(learning_data.profile_ranks[model_index][profile_index][criterion_index] == lowest_destination_rank);
    return;
  }

  Array1D<Host, unsigned> host_destination_ranks(max_destinations_count, uninitialized);
  unsigned actual_destinations_count = 0;
  if (highest_destination_rank - lowest_destination_rank >= max_destinations_count) {
    for (unsigned destination_index = 0; destination_index != max_destinations_count; ++destination_index) {
      host_destination_ranks[destination_index] = std::uniform_int_distribution<unsigned>(lowest_destination_rank, highest_destination_rank)(learning_data.urbgs[model_index]);
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

  copy(host_destination_ranks, ref(gpu_learning_data.destination_ranks[model_index]));
  gpu_learning_data.desirabilities[model_index].fill_with_zeros();
  Grid grid = grid::make(host_learning_data.alternatives_count, actual_destinations_count);
  compute_move_desirabilities__kernel<<<LOVE_CONFIG(grid)>>>(
    gpu_learning_data.performance_ranks,
    gpu_learning_data.assignments,
    gpu_learning_data.weights,
    gpu_learning_data.profile_ranks,
    model_index,
    profile_index,
    criterion_index,
    actual_destinations_count,
    gpu_learning_data.destination_ranks[model_index],
    ref(gpu_learning_data.desirabilities[model_index]));
  check_last_cuda_error_sync_stream(cudaStreamDefault);

  apply_best_move__kernel<<<1, 1>>>(
    ref(gpu_learning_data.profile_ranks),
    model_index,
    profile_index,
    criterion_index,
    actual_destinations_count,
    gpu_learning_data.destination_ranks[model_index],
    gpu_learning_data.desirabilities[model_index],
    std::uniform_real_distribution<float>(0, 1)(host_learning_data.urbgs[model_index]));
  check_last_cuda_error_sync_stream(cudaStreamDefault);

  // @todo(Performance, later) Can we group this copying somehow?
  // Currently we copy just one float from device memory to host memory
  // (because just one float is potentialy modified by 'apply_best_move__kernel',
  // and we need it back on the device for the next iteration)

  // Lov-e-CUDA doesn't provide a way to copy scalars, so we're back to the basics, using cudaMemcpy directly and doing pointer arithmetic.
  check_cuda_error(cudaMemcpy(
    host_learning_data.profile_ranks[model_index][profile_index].data() + criterion_index,
    gpu_learning_data.profile_ranks[model_index][profile_index].data() + criterion_index,
    1 * sizeof(unsigned),
    cudaMemcpyDeviceToHost));
}

}  // namespace lincs
