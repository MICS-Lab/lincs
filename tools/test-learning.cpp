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

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  auto [model, accuracy] = ppl::learning::learn_from(random, learning_set);
  model.save_to(std::cout);

  return accuracy == learning_set.alternatives_count ? 0 : 1;
}
