// Copyright 2021 Vincent Jacques

#include "learning.hpp"

#include <vector>
#include <algorithm>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "improve-weights.hpp"
#include "median-and-max.hpp"
#include "stopwatch.hpp"


namespace ppl {

Learning::Learning(const io::LearningSet& learning_set) :
  _host_domain(Domain<Host>::make(learning_set)),
  _target_accuracy(learning_set.alternatives_count),
  _use_gpu(UseGpu::Auto)
{}

template<typename Iterator>
void initialize_models(Models<Host>* models, Iterator model_indexes_begin, const Iterator model_indexes_end) {
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

std::vector<uint> partition_models_by_accuracy(const uint models_count, const Models<Host>& models) {
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

std::function<bool(uint, uint)> make_terminate(
    std::optional<uint> max_iterations,
    uint target_accuracy,
    std::optional<std::chrono::steady_clock::duration> max_duration
) {
  std::function<bool(uint, uint)> r = [target_accuracy](uint, uint accuracy) {
    return accuracy >= target_accuracy;
  };

  if (max_iterations) {
    uint max_it = *max_iterations;
    r = [max_it, r](uint iteration, uint accuracy) {
      return iteration >= max_it || r(iteration, accuracy);
    };
  }

  if (max_duration) {
    auto max_time = std::chrono::steady_clock::now() + *max_duration;
    r = [max_time, r](uint iteration, uint accuracy) {
      return r(iteration, accuracy) || std::chrono::steady_clock::now() >= max_time;
    };
  }

  return r;
}

bool use_gpu(Learning::UseGpu use) {
  switch (use) {
    case Learning::UseGpu::Force:
      return true;
    case Learning::UseGpu::Forbid:
      return false;
    case Learning::UseGpu::Auto:
    default:
      // @todo Detect GPU and return true only if it's usable
      return true;
  }
}

Learning::Result Learning::perform() const {
  STOPWATCH("Learning::perform");

  const std::function<bool(uint, uint)> terminate = make_terminate(_max_iterations, _target_accuracy, _max_duration);

  RandomSource random;
  const uint random_seed = _random_seed ? *_random_seed : std::random_device()();

  const uint models_count = _models_count ? *_models_count : 9;  // @todo Decide on a good default value

  auto host_models = Models<Host>::make(_host_domain, models_count);
  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  initialize_models(&host_models, model_indexes.begin(), model_indexes.end());

  uint best_accuracy = 0;

  if (use_gpu(_use_gpu)) {
    random.init_for_device(random_seed);

    auto device_domain = _host_domain.clone_to<Device>();
    auto device_models = host_models.clone_to<Device>(device_domain);

    for (int i = 0; !terminate(i, best_accuracy); ++i) {
      STOPWATCH("Learning::perform iteration");

      improve_weights(&host_models);
      replicate_weights(host_models, &device_models);
      improve_profiles(random, &device_models);
      replicate_profiles(device_models, &host_models);
      model_indexes = partition_models_by_accuracy(models_count, host_models);
      initialize_models(&host_models, model_indexes.begin(), model_indexes.begin() + models_count / 2);

      best_accuracy = get_accuracy(host_models, model_indexes.back());
      std::cerr << "After iteration n°" << i << ": best accuracy = " <<
        best_accuracy << "/" << _host_domain.get_view().learning_alternatives_count << std::endl;
    }
  } else {
    random.init_for_host(random_seed);

    for (int i = 0; !terminate(i, best_accuracy); ++i) {
      STOPWATCH("Learning::perform iteration");

      improve_weights(&host_models);
      improve_profiles(random, &host_models);
      model_indexes = partition_models_by_accuracy(models_count, host_models);
      initialize_models(&host_models, model_indexes.begin(), model_indexes.begin() + models_count / 2);

      best_accuracy = get_accuracy(host_models, model_indexes.back());
      std::cerr << "After iteration n°" << i << ": best accuracy = " <<
        best_accuracy << "/" << _host_domain.get_view().learning_alternatives_count << std::endl;
    }
  }

  return Result(host_models.unmake_one(model_indexes.back()), best_accuracy);
}

}  // namespace ppl
