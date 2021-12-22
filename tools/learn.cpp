// Copyright 2021 Vincent Jacques

#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>

#include <chrones.hpp>
#include <CLI11.hpp>

#include "../library/learning.hpp"
#include "../library/dump-intermediate-models.hpp"


CHRONABLE("learn")

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

  std::optional<uint> max_duration;
  app.add_option("--max-duration-seconds", max_duration);

  std::optional<uint> models_count;
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

  std::optional<uint> target_accuracy;
  if (target_accuracy_percentage)
    target_accuracy = std::ceil(*target_accuracy_percentage * learning_set.alternatives_count / 100);

  ppl::Learning learning(learning_set);

  if (target_accuracy) learning.set_target_accuracy(*target_accuracy);
  if (max_iterations) learning.set_max_iterations(*max_iterations);
  if (max_duration) learning.set_max_duration(std::chrono::seconds(*max_duration));

  if (models_count) learning.set_models_count(*models_count);
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
