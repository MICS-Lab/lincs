// Copyright 2021-2022 Vincent Jacques

#include "desirability.hpp"

#include "../assign.hpp"


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
  const uint alt_index = grid::x();
  assert(alt_index < models.domain.learning_alternatives_count + grid::blockDim.x);

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

    Grid grid = grid::make(models.domain.learning_alternatives_count);
    compute_move_desirability__kernel<<<LOVE_CONFIG(grid)>>>(
        models, model_index, profile_index, criterion_index, destination, desirability);
    check_last_cuda_error();

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

}  // namespace ppl
