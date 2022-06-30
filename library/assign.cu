// Copyright 2021-2022 Vincent Jacques

#include "assign.hpp"

#include <chrones.hpp>

#include "cuda-utils.hpp"


namespace ppl {

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
  uint accuracy = 0;

  ModelsView models_view = models.get_view();
  for (uint alt_index = 0; alt_index != models_view.domain.learning_alternatives_count; ++alt_index) {
    if (is_correctly_assigned(models_view, model_index, alt_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

__global__ void get_accuracy__kernel(ModelsView models, const uint model_index, uint* const accuracy) {
  const uint alt_index = grid::x();
  assert(alt_index < models.domain.learning_alternatives_count + grid::blockDim.x);

  if (alt_index < models.domain.learning_alternatives_count) {
    if (is_correctly_assigned(models, model_index, alt_index)) {
      atomicInc(accuracy, models.domain.learning_alternatives_count);
    }
  }
}

uint get_accuracy(const Models<Device>& models, const uint model_index) {
  CHRONE();

  uint* device_accuracy = Device::alloc<uint>(1);
  cudaMemset(device_accuracy, 0, sizeof(uint));
  check_last_cuda_error_no_sync();

  ModelsView models_view = models.get_view();
  Grid grid = grid::make(models_view.domain.learning_alternatives_count);
  get_accuracy__kernel<<<LOVE_CONFIG(grid)>>>(models_view, model_index, device_accuracy);
  check_last_cuda_error();

  uint host_accuracy;
  From<Device>::To<Host>::copy(1, device_accuracy, &host_accuracy);  // NOLINT(build/include_what_you_use)
  Device::free(device_accuracy);
  return host_accuracy;
}

}  // namespace ppl
