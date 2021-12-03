// Copyright 2021 Vincent Jacques

#include <iostream>
#include <fstream>

#include "../library/learning.hpp"
#include "../library/stopwatch.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " LEARNING_SET.txt\n";
  exit(1);
}

int main(int argc, char* argv[]) {
  STOPWATCH("test-improve-weights");

  if (argc != 2) usage(argv[0]);

  std::ifstream learning_set_file(argv[1]);
  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);

  auto result = ppl::learning::Learning(learning_set)
    .set_max_iterations(6)
    .set_target_accuracy(learning_set.alternatives_count)
    .set_random_seed(42)
    .set_models_count(15)
    .perform();

  result.best_model.save_to(std::cout);

  return result.best_model_accuracy == learning_set.alternatives_count ? 0 : 1;
}
