// Copyright 2021-2022 Vincent Jacques

#include "heuristic-for-accuracy-midpoint-candidates.hpp"

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
uint find_smallest_index_above(const MatrixView1D<const float>& m, const uint size, const float target) {
  assert(size > 0);
  assert(m[0] <= target && target <= m[size - 1]);

  uint lo = 0;
  uint hi = size - 1;
  while (lo != hi) {
    assert(lo < hi);
    const uint mid = (lo + hi) / 2;
    assert(lo <= mid && mid < hi);
    if (m[mid] >= target) {
      hi = mid;
    } else {
      lo = mid + 1;
    }
  }

  assert(m[lo] >= target);
  assert(lo == 0 || m[lo - 1] < target);
  return lo;
}

__host__ __device__
uint find_greatest_index_below(const MatrixView1D<const float>& m, const uint size, const float target) {
  assert(size > 0);
  assert(m[0] <= target && target <= m[size - 1]);

  uint lo = 0;
  uint hi = size - 1;
  while (lo != hi) {
    assert(lo < hi);
    const uint mid = (lo + hi + 1) / 2;
    assert(lo < mid && mid <= hi);
    if (m[mid] <= target) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }

  assert(m[lo] <= target);
  assert(lo == size - 1 || m[lo + 1] > target);
  return lo;
}

namespace {

__host__ __device__
void improve_model_profile(
  RandomNumberGenerator random,
  ModelsView models,
  CandidatesView candidates,
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

  const uint lowest_candidate_index = find_smallest_index_above(
    candidates.candidates[criterion_index],
    candidates.candidates_counts[criterion_index],
    lowest_destination);
  const uint highest_candidate_index = find_greatest_index_below(
    candidates.candidates[criterion_index],
    candidates.candidates_counts[criterion_index],
    highest_destination);

  // If the difference between `lowest_candidate_index` and `highest_candidate_index`
  // is in the same order of magnitude as 64, then the current choice strategy
  // (pick and put back) is suboptimal.
  for (uint n = 0; n < 64; ++n) {
    // Map (embarrassingly parallel)
    const uint candidate_index = random.uniform_int(lowest_candidate_index, highest_candidate_index + 1);
    const float destination = candidates.candidates[criterion_index][candidate_index];
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
  CandidatesView candidates,
  const uint model_index,
  const uint profile_index,
  MatrixView1D<uint> criterion_indexes
) {
  CHRONE();

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    improve_model_profile(random, models, candidates, model_index, profile_index, criterion_indexes[crit_idx_idx]);
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
void improve_model_profiles(
  RandomNumberGenerator random,
  const ModelsView& models,
  const CandidatesView& candidates,
  const uint model_index
) {
  CHRONE();

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
    improve_model_profile(random, models, candidates, model_index, profile_index, criterion_indexes);
  }

  delete[] criterion_indexes_;
}

}  // namespace

void ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu::improve_profiles(
    std::shared_ptr<Models<Host>> models
) {
  CHRONE();

  auto models_view = models->get_view();
  auto candidates_view = _candidates->get_view();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    improve_model_profiles(_random, models_view, candidates_view, model_index);
  }
}

namespace {

__global__ void improve_profiles__kernel(RandomNumberGenerator random, ModelsView models, CandidatesView candidates) {
  const uint model_index = threadIdx.x + BLOCKDIM * blockIdx.x;
  assert(model_index < models.models_count + BLOCKDIM);

  if (model_index < models.models_count) {
    improve_model_profiles(random, models, candidates, model_index);
  }
}

}  // namespace

void ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu::improve_profiles(
    std::shared_ptr<Models<Host>> host_models
) {
  CHRONE();

  replicate_models(*host_models, _device_models.get());

  auto models_view = _device_models->get_view();
  auto candidates_view = _device_candidates->get_view();

  improve_profiles__kernel<<<CONFIG(models_view.models_count)>>>(_random, models_view, candidates_view);
  cudaDeviceSynchronize();
  checkCudaErrors();

  replicate_profiles(*_device_models, host_models.get());
}

}  // namespace ppl
