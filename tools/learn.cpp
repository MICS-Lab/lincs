// Copyright 2021-2022 Vincent Jacques

#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>

#include <chrones.hpp>
#include <CLI11.hpp>

#include "../library/improve-profiles/heuristic-for-accuracy.hpp"
#include "../library/initialize-profiles/max-power-per-criterion.hpp"
#include "../library/learning.hpp"
#include "../library/observe/dump-intermediate-models.hpp"
#include "../library/observe/report-progress.hpp"
#include "../library/optimize-weights/glop.hpp"
#include "../library/terminate/accuracy.hpp"
#include "../library/terminate/any.hpp"
#include "../library/terminate/duration.hpp"
#include "../library/terminate/iterations.hpp"


CHRONABLE("learn")

std::vector<std::shared_ptr<ppl::LearningObserver>> make_observers(
  const bool quiet,
  std::optional<std::ofstream>& intermediate_models_file
) {
  std::vector<std::shared_ptr<ppl::LearningObserver>> observers;

  if (intermediate_models_file) {
    observers.push_back(std::make_shared<ppl::DumpIntermediateModels>(*intermediate_models_file));
  }

  if (!quiet) {
    observers.push_back(std::make_shared<ppl::ReportProgress>());
  }

  return observers;
}

std::shared_ptr<ppl::ProfilesInitializationStrategy> make_profiles_initialization_strategy(
  RandomNumberGenerator random,
  const ppl::Models<Host>& models
) {
  // @todo Complete with other strategies
  return std::make_shared<ppl::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion>(
    random, models);
}

std::shared_ptr<ppl::WeightsOptimizationStrategy> make_weights_optimization_strategy() {
  // @todo Complete with other strategies
  return std::make_shared<ppl::OptimizeWeightsUsingGlop>();
}

std::shared_ptr<ppl::ProfilesImprovementStrategy> make_profiles_improvement_strategy(
  RandomNumberGenerator random,
  ppl::Models<Host>* host_models,
  std::optional<ppl::Models<Device>*> device_models
) {
  if (device_models) {
    return std::make_shared<ppl::ImproveProfilesWithAccuracyHeuristicOnGpu>(random, host_models, *device_models);
  } else {
    return std::make_shared<ppl::ImproveProfilesWithAccuracyHeuristicOnCpu>(random, host_models);
  }
}

std::shared_ptr<ppl::TerminationStrategy> make_termination_strategy(
  uint target_accuracy,
  std::optional<uint> max_iterations,
  std::optional<std::chrono::seconds> max_duration
) {
  std::vector<std::shared_ptr<ppl::TerminationStrategy>> termination_strategies;

  termination_strategies.push_back(
    std::make_shared<ppl::TerminateAtAccuracy>(target_accuracy));

  if (max_iterations) {
    termination_strategies.push_back(
      std::make_shared<ppl::TerminateAfterIterations>(*max_iterations));
  }

  if (max_duration) {
    termination_strategies.push_back(
      std::make_shared<ppl::TerminateAfterDuration>(*max_duration));
  }

  if (termination_strategies.size() == 1) {
    return termination_strategies[0];
  } else {
    return std::make_shared<ppl::TerminateOnAny>(termination_strategies);
  }
}

int main(int argc, char* argv[]) {
  CHRONE();

  CLI::App app(
    "Learn a model from the learning set taken from file LEARNING_SET.txt.\n"
    "\n"
    "The resulting model is printed on standard output.\n"
    "\n"
    "Learning is not deterministic: different runs will produce different models.\n");

  std::string learning_set_file_name;
  app.add_option("LEARNING_SET.txt", learning_set_file_name)
    ->required()
    ->check(CLI::ExistingFile);

  std::optional<float> target_accuracy_percentage;
  app.add_option("--target-accuracy", target_accuracy_percentage, "as a percentage")
    ->check(CLI::Range(0., 100.));

  std::optional<uint> max_iterations;
  app.add_option("--max-iterations", max_iterations);

  std::optional<uint> max_duration_seconds;
  app.add_option("--max-duration-seconds", max_duration_seconds);

  uint models_count = ppl::default_models_count;
  app.add_option("--models-count", models_count);

  uint random_seed = std::random_device()();
  app.add_option("--random-seed", random_seed);

  bool force_gpu = false;
  auto force_gpu_flag = app.add_flag("--force-gpu", force_gpu);

  bool forbid_gpu = false;
  app.add_flag("--forbid-gpu", forbid_gpu)
    ->excludes(force_gpu_flag);

  bool quiet = false;
  app.add_flag("--quiet", quiet, "don't show progress on standard error");

  std::optional<std::string> intermediate_models_file_name;
  app.add_option("--dump-intermediate-models", intermediate_models_file_name)
    ->check(CLI::NonexistentPath);

  CLI11_PARSE(app, argc, argv);

  std::ifstream learning_set_file(learning_set_file_name);
  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);

  const uint target_accuracy =
    target_accuracy_percentage
    ?
    std::ceil(*target_accuracy_percentage * learning_set.alternatives_count / 100)
    :
    learning_set.alternatives_count;

  // Todo (much later): use C++23's std::optional::transform
  std::optional<std::chrono::seconds> max_duration;
  if (max_duration_seconds)
    max_duration = std::chrono::seconds(*max_duration_seconds);

  auto host_domain = ppl::Domain<Host>::make(learning_set);
  auto host_models = ppl::Models<Host>::make(host_domain, models_count);

  // @todo Detect GPU...
  // and verify it's usable when force_gpu is set
  // or set use_gpu according to its usability.
  // For the now, we always use the GPU unless explicitely forbiden.
  const bool use_gpu = !forbid_gpu;

  std::optional<ppl::Domain<Device>> device_domain;
  std::optional<ppl::Models<Device>> device_models;
  std::optional<ppl::Models<Device>*> device_models_address;
  if (use_gpu) {
    device_domain = host_domain.clone_to<Device>();
    device_models = host_models.clone_to<Device>(*device_domain);
    device_models_address = &*device_models;
  }

  std::optional<std::ofstream> intermediate_models_file;
  if (intermediate_models_file_name) {
    intermediate_models_file = std::ofstream(*intermediate_models_file_name);
  }

  RandomSource random;
  random.init_for_host(random_seed);
  if (use_gpu) {
    random.init_for_device(random_seed);
  }

  auto result = ppl::perform_learning(
    &host_models,
    make_observers(quiet, intermediate_models_file),
    make_profiles_initialization_strategy(random, host_models),
    make_weights_optimization_strategy(),
    make_profiles_improvement_strategy(random, &host_models, device_models_address),
    make_termination_strategy(target_accuracy, max_iterations, max_duration));

  result.best_model.save_to(std::cout);
  if (target_accuracy_percentage && result.best_model_accuracy < target_accuracy) {
    std::cerr << "Accuracy reached ("
      << float(result.best_model_accuracy) / learning_set.alternatives_count * 100
      << "%) is below target" << std::endl;
    return 1;
  }

  return 0;
}
