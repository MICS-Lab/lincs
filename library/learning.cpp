// Copyright 2021 Vincent Jacques

#include "learning.hpp"

#include <vector>
#include <algorithm>

#include <chrones.hpp>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "optimize-weights.hpp"
#include "initialize-profiles.hpp"
#include "median-and-max.hpp"


namespace ppl {

std::vector<uint> partition_models_by_accuracy(const uint models_count, const Models<Host>& models) {
  CHRONE();

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
    r = [max_it, r](uint iteration_index, uint accuracy) {
      return iteration_index >= max_it || r(iteration_index, accuracy);
    };
  }

  if (max_duration) {
    auto max_time = std::chrono::steady_clock::now() + *max_duration;
    r = [max_time, r](uint iteration_index, uint accuracy) {
      return r(iteration_index, accuracy) || std::chrono::steady_clock::now() >= max_time;
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

// https://en.wikipedia.org/wiki/Curiously_recurring_template_pattern#Static_polymorphism
template<typename ConcreteLearningExecution>
struct LearningExecution {
  LearningExecution(
    const Domain<Host>& host_domain_,
    uint models_count_,
    std::function<bool(uint, uint)> terminate_,
    uint random_seed_,
    std::vector<std::shared_ptr<Learning::Observer>> observers_) :
      self(static_cast<ConcreteLearningExecution&>(*this)),
      models_count(models_count_),
      model_indexes(models_count, 0),
      terminate(terminate_),
      host_domain(host_domain_),
      host_models(Models<Host>::make(host_domain, models_count)),
      random_seed(random_seed_),
      random(),
      profiles_initializer(host_models),
      weights_optimizer(host_models),
      observers(observers_) {
    random.init_for_host(random_seed);
    std::iota(model_indexes.begin(), model_indexes.end(), 0);
    profiles_initializer.initialize_profiles(random, &host_models, 0, model_indexes.begin(), model_indexes.end());
  }

  Learning::Result execute() {
    CHRONE();

    uint best_accuracy = 0;

    for (int iteration_index = 0; !terminate(iteration_index, best_accuracy); ++iteration_index) {
      if (iteration_index != 0) {
        profiles_initializer.initialize_profiles(
          random, &host_models,
          iteration_index,
          model_indexes.begin(), model_indexes.begin() + models_count / 2);
      }

      weights_optimizer.optimize_weights(&host_models);
      self.improve_profiles();

      model_indexes = partition_models_by_accuracy(models_count, host_models);
      best_accuracy = get_accuracy(host_models, model_indexes.back());

      for (auto observer : observers) {
        observer->after_main_iteration(iteration_index, best_accuracy, host_models);
      }
    }

    return Learning::Result(host_models.unmake_one(model_indexes.back()), best_accuracy);
  }

 private:
  ConcreteLearningExecution& self;
  uint models_count;
  std::vector<uint> model_indexes;
  std::function<bool(uint, uint)> terminate;

 protected:
  const Domain<Host>& host_domain;
  Models<Host> host_models;
  uint random_seed;
  RandomSource random;

 private:
  ProfilesInitializer profiles_initializer;
  WeightsOptimizer weights_optimizer;
  std::vector<std::shared_ptr<Learning::Observer>> observers;
};

struct GpuLearningExecution : LearningExecution<GpuLearningExecution> {
  GpuLearningExecution(
    const Domain<Host>& host_domain,
    uint models_count,
    std::function<bool(uint, uint)> terminate,
    uint random_seed,
    std::vector<std::shared_ptr<Learning::Observer>> observers) :
      LearningExecution<GpuLearningExecution>(host_domain, models_count, terminate, random_seed, observers),
      device_domain(host_domain.clone_to<Device>()),
      device_models(host_models.clone_to<Device>(device_domain)) {
    random.init_for_device(random_seed);
  }

  void improve_profiles() {
    replicate_models(host_models, &device_models);
    profiles_improver.improve_profiles(random, &device_models);
    replicate_profiles(device_models, &host_models);
  }

 private:
  Domain<Device> device_domain;
  Models<Device> device_models;

  ProfilesImprover profiles_improver;
};

struct CpuLearningExecution : LearningExecution<CpuLearningExecution> {
  CpuLearningExecution(
    const Domain<Host>& host_domain,
    uint models_count,
    std::function<bool(uint, uint)> terminate,
    uint random_seed,
    std::vector<std::shared_ptr<Learning::Observer>> observers) :
      LearningExecution<CpuLearningExecution>(host_domain, models_count, terminate, random_seed, observers) {
  }

  void improve_profiles() {
    profiles_improver.improve_profiles(random, &host_models);
  }

 private:
  ProfilesImprover profiles_improver;
};

Learning::Result Learning::perform() const {
  CHRONE();

  const uint models_count = _models_count ? *_models_count : 9;  // @todo Decide on a good default value
  const std::function<bool(uint, uint)> terminate = make_terminate(_max_iterations, _target_accuracy, _max_duration);
  const uint random_seed = _random_seed ? *_random_seed : std::random_device()();

  if (use_gpu(_use_gpu)) {
    return GpuLearningExecution(_host_domain, models_count, terminate, random_seed, _observers).execute();
  } else {
    return CpuLearningExecution(_host_domain, models_count, terminate, random_seed, _observers).execute();
  }
}

}  // namespace ppl
