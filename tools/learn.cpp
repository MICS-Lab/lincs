// Copyright 2021 Vincent Jacques

#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>

#include <chrones.hpp>
#include <CLI11.hpp>

#include "../library/dump-intermediate-models.hpp"
#include "../library/learning.hpp"
#include "../library/terminate/accuracy.hpp"
#include "../library/terminate/any.hpp"
#include "../library/terminate/duration.hpp"
#include "../library/terminate/iterations.hpp"


CHRONABLE("learn")

std::shared_ptr<ppl::TerminationStrategy> make_termination_strategy(
  const ppl::io::LearningSet& learning_set,
  std::optional<uint> target_accuracy,
  std::optional<uint> max_iterations,
  std::optional<std::chrono::seconds> max_duration
) {
  std::vector<std::shared_ptr<ppl::TerminationStrategy>> terminate_strategies;

  if (target_accuracy) {
    terminate_strategies.push_back(
      std::make_shared<ppl::TerminateAtAccuracy>(*target_accuracy));
  } else {
    terminate_strategies.push_back(
      std::make_shared<ppl::TerminateAtAccuracy>(learning_set.alternatives_count));
  }

  if (max_iterations) {
    terminate_strategies.push_back(
      std::make_shared<ppl::TerminateAfterIterations>(*max_iterations));
  }

  if (max_duration) {
    terminate_strategies.push_back(
      std::make_shared<ppl::TerminateAfterDuration>(*max_duration));
  }

  if (terminate_strategies.size() == 1) {
    return terminate_strategies[0];
  } else {
    return std::make_shared<ppl::TerminateOnAny>(terminate_strategies);
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

  // @todo Find a good default value
  // @todo Put the default value in the `learning` module
  uint models_count = 9;
  app.add_option("--models-count", models_count);

  std::optional<uint> random_seed;
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

  // Todo (much later): use C++23's std::optional::transform
  std::optional<uint> target_accuracy;
  if (target_accuracy_percentage)
    target_accuracy = std::ceil(*target_accuracy_percentage * learning_set.alternatives_count / 100);

  // Todo (much later): use C++23's std::optional::transform
  std::optional<std::chrono::seconds> max_duration;
  if (max_duration_seconds)
    max_duration = std::chrono::seconds(*max_duration_seconds);

  auto domain = ppl::Domain<Host>::make(learning_set);
  auto models = ppl::Models<Host>::make(domain, models_count);

  ppl::Learning learning(
    domain, &models,
    make_termination_strategy(learning_set, target_accuracy, max_iterations, max_duration));

  if (random_seed) learning.set_random_seed(*random_seed);

  if (force_gpu) learning.force_using_gpu();
  if (forbid_gpu) learning.forbid_using_gpu();

  if (!quiet) learning.subscribe(std::make_shared<ppl::Learning::ProgressReporter>());

  std::optional<std::ofstream> intermediate_models_file;
  if (intermediate_models_file_name) {
    intermediate_models_file = std::ofstream(*intermediate_models_file_name);
    learning.subscribe(std::make_shared<ppl::IntermediateModelsDumper>(*intermediate_models_file));
  }

  auto result = learning.perform();
  result.best_model.save_to(std::cout);
  if (target_accuracy && result.best_model_accuracy < *target_accuracy) {
    std::cerr << "Accuracy reached ("
      << float(result.best_model_accuracy) / learning_set.alternatives_count * 100
      << "%) is below target" << std::endl;
    return 1;
  }

  return 0;
}
