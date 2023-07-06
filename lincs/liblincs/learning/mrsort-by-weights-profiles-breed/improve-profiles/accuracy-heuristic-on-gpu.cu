// Copyright 2023 Vincent Jacques

#include "accuracy-heuristic-on-gpu.hpp"
#include "../../../randomness-utils.hpp"


namespace {

typedef GridFactory2D<256, 4> grid;

__device__
unsigned get_assignment(
  const ArrayView2D<Device, const float> learning_alternatives,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const float> profiles,
  const unsigned model_index,
  const unsigned alternative_index
) {
  const unsigned criteria_count = learning_alternatives.s1();
  const unsigned categories_count = profiles.s1() + 1;

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const float alternative_value = learning_alternatives[criterion_index][alternative_index];
      const float profile_value = profiles[criterion_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += weights[criterion_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

__device__
void update_move_desirability(
  const ArrayView2D<Device, const float> learning_alternatives,
  const ArrayView1D<Device, const unsigned> learning_assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const float> profiles,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const float destination,
  const unsigned alternative_index,
  lincs::Desirability* desirability
) {
  const unsigned learning_alternatives_count = learning_alternatives.s0();
  const unsigned criteria_count = learning_alternatives.s1();

  const float current_position = profiles[criterion_index][profile_index][model_index];
  const float weight = weights[criterion_index][model_index];

  const float value = learning_alternatives[criterion_index][alternative_index];
  const unsigned learning_assignment = learning_assignments[alternative_index];
  const unsigned model_assignment = get_assignment(
    learning_alternatives,
    weights,
    profiles,
    model_index,
    alternative_index);

  float weight_at_or_above_profile = 0;
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const float alternative_value = learning_alternatives[criterion_index][alternative_index];
    const float profile_value = profiles[criterion_index][profile_index][model_index];
    if (alternative_value >= profile_value) {
      weight_at_or_above_profile += weights[criterion_index][model_index];
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
  // - weight_at_or_above_profile: \sigma
  // - weight: w_j
  // - 1: \lambda
  if (destination > current_position) {
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight < 1) {
        atomicInc(&desirability->v, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight >= 1) {
        atomicInc(&desirability->w, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight < 1) {
        atomicInc(&desirability->q, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination > value
      && value >= current_position) {
        atomicInc(&desirability->r, learning_alternatives_count);
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && destination > value
      && value >= current_position) {
        atomicInc(&desirability->t, learning_alternatives_count);
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        atomicInc(&desirability->v, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight < 1) {
        atomicInc(&desirability->w, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        atomicInc(&desirability->q, learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination <= value
      && value < current_position) {
        atomicInc(&desirability->r, learning_alternatives_count);
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && destination < value
      && value <= current_position) {
        atomicInc(&desirability->t, learning_alternatives_count);
    }
  }
}

// @todo investigate how sharing preliminary computations done in all threads could improve perf
__global__
void compute_move_desirabilities__kernel(
  const ArrayView2D<Device, const float> learning_alternatives,
  const ArrayView1D<Device, const unsigned> learning_assignments,
  const ArrayView2D<Device, const float> weights,
  const ArrayView3D<Device, const float> profiles,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const ArrayView1D<Device, const float> destinations,
  ArrayView1D<Device, lincs::Desirability> desirabilities
) {
  const unsigned alt_index = grid::x();
  assert(alt_index < learning_alternatives.s0() + grid::blockDim.x);
  const unsigned destination_index = grid::y();
  assert(destination_index < destinations.s0() + grid::blockDim.y);

  // Map (embarrassingly parallel)
  if (alt_index < learning_alternatives.s0() && destination_index < destinations.s0()) {
    update_move_desirability(
      learning_alternatives,
      learning_assignments,
      weights,
      profiles,
      model_index,
      profile_index,
      criterion_index,
      destinations[destination_index],
      alt_index,
      &desirabilities[destination_index]);
  }
}

__global__
void apply_best_move__kernel(
  const ArrayView3D<Device, float> profiles,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const ArrayView1D<Device, const float> destinations,
  const ArrayView1D<Device, const lincs::Desirability> desirabilities,
  const float desirability_threshold
) {
  // Single-key reduce
  // Could maybe be parallelized using divide and conquer? Or atomic compare-and-swap?
  float best_destination = destinations[0];
  float best_desirability = desirabilities[0].value();
  for (unsigned destination_index = 1; destination_index < destinations.s0(); ++destination_index) {
    const float destination = destinations[destination_index];
    const float desirability = desirabilities[destination_index].value();

    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  if (best_desirability >= desirability_threshold) {
    profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

}  // namespace

namespace lincs {

ImproveProfilesWithAccuracyHeuristicOnGpu::GpuLearningData ImproveProfilesWithAccuracyHeuristicOnGpu::GpuLearningData::make(const LearningData& learning_data) {
  return {
    learning_data.categories_count,
    learning_data.criteria_count,
    learning_data.learning_alternatives_count,
    learning_data.learning_alternatives.template clone_to<Device>(),
    learning_data.learning_assignments.template clone_to<Device>(),
    learning_data.models_count,
    Array2D<Device, float>(learning_data.criteria_count, learning_data.models_count, uninitialized),
    Array3D<Device, float>(learning_data.criteria_count, (learning_data.categories_count - 1), learning_data.models_count, uninitialized),
    Array2D<Device, Desirability>(learning_data.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::destinations_count, uninitialized),
    Array2D<Device, float>(learning_data.models_count, ImproveProfilesWithAccuracyHeuristicOnGpu::destinations_count, uninitialized),
  };
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles() {
  // Get optimized weights
  copy(host_learning_data.weights, ref(gpu_learning_data.weights));
  // Get (re-)initialized profiles
  copy(host_learning_data.profiles, ref(gpu_learning_data.profiles));

  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != gpu_learning_data.models_count; ++model_index) {
    improve_model_profiles(model_index);
  }

  // Set improved profiles
  copy(gpu_learning_data.profiles, ref(host_learning_data.profiles));
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(gpu_learning_data.criteria_count, uninitialized);
  // Not worth parallelizing because learning_data.criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != gpu_learning_data.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned profile_index = 0; profile_index != gpu_learning_data.categories_count - 1; ++profile_index) {
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
  for (unsigned crit_idx_idx = 0; crit_idx_idx != gpu_learning_data.criteria_count; ++crit_idx_idx) {
    improve_model_profile(model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index
) {
  // WARNING: We're assuming all criteria have values in [0, 1]
  const float lowest_destination =
    profile_index == 0 ? 0. :
    host_learning_data.profiles[criterion_index][profile_index - 1][model_index];
  const float highest_destination =
    profile_index == host_learning_data.categories_count - 2 ? 1. :
    host_learning_data.profiles[criterion_index][profile_index + 1][model_index];

  if (lowest_destination == highest_destination) {
    assert(host_learning_data.profiles[criterion_index][profile_index][model_index] == lowest_destination);
    return;
  }

  Array1D<Host, float> host_destinations(destinations_count, uninitialized);
  for (unsigned destination_index = 0; destination_index != destinations_count; ++destination_index) {
    float destination = highest_destination;
    // By specification, std::uniform_real_distribution should never return its highest value,
    // but "most existing implementations have a bug where they may occasionally" return it,
    // so we work around that bug by calling it again until it doesn't.
    // Ref: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    while (destination == highest_destination) {
      destination = std::uniform_real_distribution<float>(lowest_destination, highest_destination)(host_learning_data.urbgs[model_index]);
    }
    host_destinations[destination_index] = destination;
  }

  copy(host_destinations, ref(gpu_learning_data.destinations[model_index]));
  gpu_learning_data.desirabilities[model_index].fill_with_zeros();
  Grid grid = grid::make(gpu_learning_data.learning_alternatives_count, destinations_count);
  compute_move_desirabilities__kernel<<<LOVE_CONFIG(grid)>>>(
    gpu_learning_data.learning_alternatives,
    gpu_learning_data.learning_assignments,
    gpu_learning_data.weights,
    gpu_learning_data.profiles,
    model_index,
    profile_index,
    criterion_index,
    gpu_learning_data.destinations[model_index],
    ref(gpu_learning_data.desirabilities[model_index]));
  check_last_cuda_error_sync_stream(cudaStreamDefault);

  apply_best_move__kernel<<<1, 1>>>(
    ref(gpu_learning_data.profiles),
    model_index,
    profile_index,
    criterion_index,
    gpu_learning_data.destinations[model_index],
    gpu_learning_data.desirabilities[model_index],
    std::uniform_real_distribution<float>(0, 1)(host_learning_data.urbgs[model_index]));
  check_last_cuda_error_sync_stream(cudaStreamDefault);

  // @todo Double-check and document why we don't need [model_index] here
  copy(gpu_learning_data.profiles[criterion_index][profile_index], host_learning_data.profiles[criterion_index][profile_index]);
}

}  // namespace lincs
