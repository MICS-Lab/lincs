// Copyright 2021 Vincent Jacques

#include "initialize.hpp"

#include "stopwatch.hpp"


namespace ppl {

void ModelsInitializer::initialize(
  Models<Host>* models,
  std::vector<uint>::const_iterator model_indexes_begin,
  const std::vector<uint>::const_iterator model_indexes_end
) {
  STOPWATCH("ModelsInitializer::initialize");

  ModelsView models_view = models->get_view();

  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const uint model_index = *model_indexes_begin;

    // @todo Implement as described in Sobrie's thesis
    for (uint profile_index = 0; profile_index != models_view.domain.categories_count - 1; ++profile_index) {
      const float value = static_cast<float>(profile_index + 1) / models_view.domain.categories_count;
      for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
        models_view.profiles[crit_index][profile_index][model_index] = value;
      }
    }
    for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
      models_view.weights[crit_index][model_index] = 2. / models_view.domain.criteria_count;
    }
  }
}

}  // namespace ppl
