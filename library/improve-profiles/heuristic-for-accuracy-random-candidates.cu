// Copyright 2021-2022 Vincent Jacques

#include "heuristic-for-accuracy-random-candidates.hpp"

#include <algorithm>
#include <utility>
#include <cassert>
#include <random>

#include <chrones.hpp>

#include "../assign.hpp"
#include "../cuda-utils.hpp"
#include "desirability.hpp"

namespace ppl {

__host__ __device__
void improve_model_profile(
  RandomNumberGenerator random,
  ModelsView models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index
) {
  CHRONE();

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
  ArrayView1D<Anywhere, uint> criterion_indexes
) {
  CHRONE();

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
void shuffle(RandomNumberGenerator random, ArrayView1D<Anywhere, T> m) {
  for (uint i = 0; i != m.s0(); ++i) {
    swap(m[i], m[random.uniform_int(0, m.s0())]);
  }
}

__host__ __device__
void improve_model_profiles(RandomNumberGenerator random, const ModelsView& models, const uint model_index) {
  CHRONE();

  #ifdef __CUDA_ARCH__
    typedef Device CurrentSpace;
  #else
    typedef Host CurrentSpace;
  #endif

  Array1D<CurrentSpace, uint> criterion_indexes(models.domain.criteria_count, uninitialized);
  // Not worth parallelizing because models.domain.criteria_count is typically small
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint profile_index = 0; profile_index != models.domain.categories_count - 1; ++profile_index) {
    shuffle<uint>(random, criterion_indexes);
    improve_model_profile(random, models, model_index, profile_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles(std::shared_ptr<Models<Host>> models) {
  CHRONE();

  auto models_view = models->get_view();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    improve_model_profiles(_random, models_view, model_index);
  }
}

__global__ void improve_profiles__kernel(RandomNumberGenerator random, ModelsView models) {
  const uint model_index = grid::x();
  assert(model_index < models.models_count + grid::blockDim.x);

  if (model_index < models.models_count) {
    improve_model_profiles(random, models, model_index);
  }
}

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles(std::shared_ptr<Models<Host>> host_models) {
  CHRONE();

  replicate_models(*host_models, _device_models.get());

  auto models_view = _device_models->get_view();

  Grid grid = grid::make(models_view.models_count);
  improve_profiles__kernel<<<LOVE_CONFIG(grid)>>>(_random, models_view);
  check_last_cuda_error();

  replicate_profiles(*_device_models, host_models.get());
}

}  // namespace ppl
