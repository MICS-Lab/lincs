// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <algorithm>
#include <utility>
#include <cassert>
#include <random>

#include "cuda-utils.hpp"
#include "stopwatch.hpp"
#include "randomness.hpp"
#include "assign.hpp"


namespace ppl {

__host__ __device__
void increment(
    uint* i,
    uint
    #ifdef __CUDA_ARCH__
    max
    #endif
) {
  #ifdef __CUDA_ARCH__
  atomicInc(i, max);
  #else
  ++*i;
  #endif
}

__host__ __device__
void update_move_desirability(
  const ModelsView& models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination,
  const uint alt_index,
  Desirability* desirability
) {
  const float current_position = models.profiles[criterion_index][profile_index][model_index];
  const float weight = models.weights[criterion_index][model_index];

  const float value = models.domain.learning_alternatives[criterion_index][alt_index];
  const uint learning_assignment = models.domain.learning_assignments[alt_index];
  const uint model_assignment = get_assignment(models, model_index, alt_index);

  // @todo Factorize with get_assignment
  float weight_at_or_above_profile = 0;
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
    const float profile_value = models.profiles[crit_index][profile_index][model_index];
    if (alternative_value >= profile_value) {
      weight_at_or_above_profile += models.weights[crit_index][model_index];
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
        increment(&(desirability->v), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight >= 1) {
        increment(&(desirability->w), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight < 1) {
        increment(&(desirability->q), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination > value
      && value >= current_position) {
        increment(&(desirability->r), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && destination > value
      && value >= current_position) {
        increment(&(desirability->t), models.domain.learning_alternatives_count);
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        increment(&(desirability->v), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight < 1) {
        increment(&(desirability->w), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        increment(&(desirability->q), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination <= value
      && value < current_position) {
        increment(&(desirability->r), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && destination < value
      && value <= current_position) {
        increment(&(desirability->t), models.domain.learning_alternatives_count);
    }
  }
}

__global__ void compute_move_desirability__kernel(
  const ModelsView models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination,
  Desirability* desirability
) {
  const uint alt_index = threadIdx.x + BLOCKDIM * blockIdx.x;
  assert(alt_index < models.domain.learning_alternatives_count + BLOCKDIM);

  if (alt_index < models.domain.learning_alternatives_count) {
    update_move_desirability(
      models, model_index, profile_index, criterion_index, destination, alt_index, desirability);
  }
}

__host__ __device__
Desirability compute_move_desirability(
  const ModelsView& models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination
) {
  #ifdef __CUDA_ARCH__
    Desirability* desirability = new Desirability;

    compute_move_desirability__kernel<<<CONFIG(models.domain.learning_alternatives_count)>>>(
        models, model_index, profile_index, criterion_index, destination, desirability);
    cudaDeviceSynchronize();
    checkCudaErrors();

    Desirability d = *desirability;
    delete desirability;
  #else
    Desirability d;
    for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
      update_move_desirability(
        models, model_index, profile_index, criterion_index, destination, alt_index, &d);
    }
  #endif

  return d;
}

__host__ __device__
void improve_model_profile(
  RandomNumberGenerator random,
  ModelsView models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index
) {
  // WARNING: We're assuming all criteria have values in [0, 1]
  // @todo Can we relax this assumption?
  // This is consistent with our comment in the header file, but slightly less generic than Sobrie's thesis
  const float lowest_destination =
    profile_index == 0 ? 0. :
    models.profiles[criterion_index][profile_index - 1][model_index];
  const float highest_destination =
    profile_index == models.domain.categories_count - 2 ? 1. :
    models.profiles[criterion_index][profile_index + 1][model_index];

  float best_destination = models.profiles[criterion_index][profile_index][model_index];
  float best_desirability = Desirability().value();

  if (lowest_destination == highest_destination) {
    assert(best_destination == lowest_destination);
    return;
  }

  // Not sure about this part: we're considering an arbitrary number of possible moves as described in
  // Mousseau's prez-mics-2018(8).pdf, but:
  //  - this is wasteful when there are fewer alternatives in the interval
  //  - this is not strictly consistent with, albeit much simpler than, Sobrie's thesis
  // @todo Ask Vincent Mousseau about the following:
  // We could consider only a finite set of values for b_j described as follows:
  // - sort all the 'a_j's
  // - compute all midpoints between two successive 'a_j'
  // - add two extreme values (0 and 1, or above the greatest a_j and below the smallest a_j)
  // Then instead of taking a random values in [lowest_destination, highest_destination],
  // we'd take a random subset of the intersection of these midpoints with that interval.
  for (uint n = 0; n < 64; ++n) {
    // Map (embarrassingly parallel)
    const float destination = random.uniform_float(lowest_destination, highest_destination);
    const float desirability = compute_move_desirability(
      models, model_index, profile_index, criterion_index, destination).value();
    // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  // @todo Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (random.uniform_float(0, 1) <= best_desirability) {
    models.profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

__host__ __device__
void improve_model_profile(
  RandomNumberGenerator random,
  ModelsView models,
  const uint model_index,
  const uint profile_index,
  MatrixView1D<uint> criterion_indexes
) {
  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    improve_model_profile(random, models, model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

template<typename T>
__host__ __device__
void swap(T& a, T& b) {
  T c = a;
  a = b;
  b = c;
}

template<typename T>
__host__ __device__
void shuffle(RandomNumberGenerator random, MatrixView1D<T> m) {
  for (uint i = 0; i != m.s0(); ++i) {
    swap(m[i], m[random.uniform_int(0, m.s0())]);
  }
}

__host__ __device__
void improve_model_profiles(RandomNumberGenerator random, const ModelsView& models, const uint model_index) {
  uint* criterion_indexes_ = new uint[models.domain.criteria_count];
  MatrixView1D<uint> criterion_indexes(models.domain.criteria_count, criterion_indexes_);
  // Not worth parallelizing because models.domain.criteria_count is typically small
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint profile_index = 0; profile_index != models.domain.categories_count - 1; ++profile_index) {
    shuffle(random, criterion_indexes);
    improve_model_profile(random, models, model_index, profile_index, criterion_indexes);
  }

  delete[] criterion_indexes_;
}

__host__ __device__
void improve_profiles(RandomNumberGenerator random, const ModelsView& models) {
  // Embarrassingly parallel
  for (uint model_index = 0; model_index != models.models_count; ++model_index) {
    improve_model_profiles(random, models, model_index);
  }
}

void improve_profiles(const RandomSource& random, Models<Host>* models) {
  STOPWATCH("improve_profiles (Host)");

  improve_profiles(random, models->get_view());
}

__global__ void improve_profiles__kernel(RandomNumberGenerator random, ModelsView models) {
  assert(blockIdx.x == 0);
  assert(threadIdx.x == 0);
  improve_profiles(random, models);
}

void improve_profiles(const RandomSource& random, Models<Device>* models) {
  STOPWATCH("improve_profiles (Device)");

  improve_profiles__kernel<<<1, 1>>>(random, models->get_view());
  cudaDeviceSynchronize();
  checkCudaErrors();
}

}  // namespace ppl
