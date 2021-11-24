// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <algorithm>
#include <utility>
#include <cassert>
#include <random>

#include "cuda-utils.hpp"
#include "stopwatch.hpp"
#include "randomness.hpp"


namespace ppl::improve_profiles {

template<typename Space>
Domain<Space>::Domain(
  const uint categories_count_,
  const uint criteria_count_,
  const uint learning_alternatives_count_,
  float* learning_alternatives_,
  uint* learning_assignments_) :
    categories_count(categories_count_),
    criteria_count(criteria_count_),
    learning_alternatives_count(learning_alternatives_count_),
    learning_alternatives(learning_alternatives_),
    learning_assignments(learning_assignments_) {}

template<>
Domain<Host> Domain<Host>::make(const io::LearningSet& learning_set) {
  assert(learning_set.is_valid());

  float* alternatives_ = alloc_host<float>(learning_set.criteria_count * learning_set.alternatives_count);
  MatrixView2D<float> alternatives(learning_set.criteria_count, learning_set.alternatives_count, alternatives_);
  uint* assignments_ = alloc_host<uint>(learning_set.alternatives_count);
  MatrixView1D<uint> assignments(learning_set.alternatives_count, assignments_);

  for (uint alt_index = 0; alt_index != learning_set.alternatives_count; ++alt_index) {
    const io::ClassifiedAlternative& alt = learning_set.alternatives[alt_index];

    for (uint crit_index = 0; crit_index != learning_set.criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria_values[crit_index];
    }

    assignments[alt_index] = alt.assigned_category;
  }

  return Domain(
    learning_set.categories_count,
    learning_set.criteria_count,
    learning_set.alternatives_count,
    alternatives_,
    assignments_);
}

template<typename Space>
Domain<Space>::~Domain() {
  Space::free(learning_alternatives);
  Space::free(learning_assignments);
}

template<typename Space>
DomainView Domain<Space>::get_view() const {
  return {
    categories_count,
    criteria_count,
    learning_alternatives_count,
    MatrixView2D<const float>(criteria_count, learning_alternatives_count, learning_alternatives),
    MatrixView1D<const uint>(learning_alternatives_count, learning_assignments),
  };
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
  const Domain<Space>& domain_,
  const uint models_count_,
  float* weights_,
  float* profiles_) :
    domain(domain_),
    models_count(models_count_),
    weights(weights_),
    profiles(profiles_) {}

template<>
Models<Host> Models<Host>::make(const Domain<Host>& domain, const std::vector<io::Model>& models) {
  DomainView domain_view = domain.get_view();
  const uint models_count = models.size();
  float* weights_ = alloc_host<float>(domain_view.criteria_count * models_count);
  MatrixView2D<float> weights(domain_view.criteria_count, models_count, weights_);
  float* profiles_ = alloc_host<float>(domain_view.criteria_count * (domain_view.categories_count - 1) * models_count);
  MatrixView3D<float> profiles(domain_view.criteria_count, domain_view.categories_count - 1, models_count, profiles_);

  for (uint model_index = 0; model_index != models_count; ++model_index) {
    const io::Model& model = models[model_index];
    assert(model.is_valid());

    for (uint crit_index = 0; crit_index != domain_view.criteria_count; ++crit_index) {
      weights[crit_index][model_index] = model.weights[crit_index];
    }

    for (uint cat_index = 0; cat_index != domain_view.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      for (uint crit_index = 0; crit_index != domain_view.criteria_count; ++crit_index) {
        profiles[crit_index][cat_index][model_index] = category_profile[crit_index];
      }
    }
  }

  return Models(domain, models_count, weights_, profiles_);
}

template<typename Space>
Models<Space>::~Models() {
  Space::free(weights);
  Space::free(profiles);
}

template<typename Space>
ModelsView Models<Space>::get_view() const {
  DomainView domain_view = domain.get_view();
  return {
    domain_view,
    models_count,
    MatrixView2D<float>(domain_view.criteria_count, models_count, weights),
    MatrixView3D<float>(domain_view.criteria_count, domain_view.categories_count - 1, models_count, profiles),
  };
}

template class Models<Host>;
template class Models<Device>;

__host__ __device__
uint get_assignment(const ModelsView& models, const uint model_index, const uint alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index < models.models_count);
  assert(alternative_index < models.domain.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (uint category_index = models.domain.categories_count - 1; category_index != 0; --category_index) {
    const uint profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alternative_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

uint get_assignment(const Models<Host>& models, const uint model_index, const uint alternative_index) {
  return get_assignment(models.get_view(), model_index, alternative_index);
}

__host__ __device__
bool is_correctly_assigned(
    const ModelsView& models,
    const uint model_index,
    const uint alternative_index) {
  const uint expected_assignment = models.domain.learning_assignments[alternative_index];
  const uint actual_assignment = get_assignment(models, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

uint get_accuracy(const Models<Host>& models, const uint model_index) {
  STOPWATCH("get_accuracy (Host)");

  uint accuracy = 0;

  ModelsView models_view = models.get_view();
  for (uint alt_index = 0; alt_index != models_view.domain.learning_alternatives_count; ++alt_index) {
    if (is_correctly_assigned(models_view, model_index, alt_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

#define BLOCKDIM 512
#define CONFIG(size) (size) / BLOCKDIM + ((size) % BLOCKDIM ? 1 : 0), BLOCKDIM

__global__ void get_accuracy__kernel(ModelsView models, const uint model_index, uint* const accuracy) {
  const uint alt_index = threadIdx.x + BLOCKDIM * blockIdx.x;
  assert(alt_index < models.domain.learning_alternatives_count + BLOCKDIM);

  if (alt_index < models.domain.learning_alternatives_count) {
    if (is_correctly_assigned(models, model_index, alt_index)) {
      atomicInc(accuracy, models.domain.learning_alternatives_count);
    }
  }
}

uint get_accuracy(const Models<Device>& models, const uint model_index) {
  STOPWATCH("get_accuracy (Device)");

  uint* device_accuracy = alloc_device<uint>(1);
  cudaMemset(device_accuracy, 0, sizeof(uint));
  checkCudaErrors();

  ModelsView models_view = models.get_view();
  get_accuracy__kernel<<<CONFIG(models_view.domain.learning_alternatives_count)>>>(
    models_view, model_index, device_accuracy);
  cudaDeviceSynchronize();
  checkCudaErrors();

  uint host_accuracy;
  copy_device_to_host(1, device_accuracy, &host_accuracy);
  free_device(device_accuracy);
  return host_accuracy;
}

__host__ __device__
void increment(uint* i, uint max) {
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
    // Map (embarassigly parallel)
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
  // Embarassingly parallel
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

}  // namespace ppl::improve_profiles
