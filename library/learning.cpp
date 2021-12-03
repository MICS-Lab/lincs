// Copyright 2021 Vincent Jacques

#include "learning.hpp"

#include <vector>
#include <algorithm>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "improve-weights.hpp"
#include "median-and-max.hpp"
#include "stopwatch.hpp"


namespace ppl::learning {

Learning::Learning(const io::LearningSet& learning_set) : _host_domain(Domain<Host>::make(learning_set)) {}

template<typename Iterator>
void initialize_models(ppl::Models<Host>* models, Iterator model_indexes_begin, const Iterator model_indexes_end) {
  STOPWATCH("initialize_models");

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

std::vector<uint> partition_models_by_accuracy(const uint models_count, const ppl::Models<Host>& models) {
  std::vector<uint> accuracies(models_count, 0);
  for (uint model_index = 0; model_index != models_count; ++model_index) {
    accuracies[model_index] = get_accuracy(models, model_index);
  }
  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [&accuracies](uint left_model_index, uint right_model_index) {
      return accuracies[left_model_index] < accuracies[right_model_index];
    });
  return model_indexes;
}

Learning::Result Learning::perform() const {
  STOPWATCH("Learning::perform");

  RandomSource random;
  random.init_for_host(*_random_seed);
  random.init_for_device(*_random_seed);

  auto device_domain = _host_domain.clone_to<Device>();

  const uint models_count = *_models_count;

  auto host_models = ppl::Models<Host>::make(_host_domain, models_count);
  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  initialize_models(&host_models, model_indexes.begin(), model_indexes.end());
  auto device_models = host_models.clone_to<Device>(device_domain);

  uint best_accuracy = 0;
  for (int i = 0; i != _max_iterations && best_accuracy < _target_accuracy; ++i) {
    STOPWATCH("Learning::perform iteration");

    improve_weights::improve_weights(&host_models);
    replicate_weights(host_models, &device_models);
    improve_profiles::improve_profiles(random, &device_models);
    replicate_profiles(device_models, &host_models);
    model_indexes = partition_models_by_accuracy(models_count, host_models);
    initialize_models(&host_models, model_indexes.begin(), model_indexes.begin() + models_count / 2);

    best_accuracy = get_accuracy(host_models, model_indexes.back());
    std::cerr << "After iteration n°" << i << ": best accuracy = " <<
      best_accuracy << "/" << _host_domain.get_view().learning_alternatives_count << std::endl;
  }

  return { host_models.unmake_one(model_indexes.back()), best_accuracy };
}

}  // namespace ppl::learning
