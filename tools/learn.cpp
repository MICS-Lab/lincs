// Copyright 2021 Vincent Jacques

#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>

#include "../library/learning.hpp"
#include "../library/stopwatch.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " LEARNING_SET.txt\n";
  exit(1);
}

int main(int argc, char* argv[]) {
  STOPWATCH("learn");

  if (argc != 2) usage(argv[0]);

  std::ifstream learning_set_file(argv[1]);
  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);
  ppl::learning::Learning learning(learning_set);
  learning.set_max_duration(std::chrono::seconds(10));
  learning.perform().best_model.save_to(std::cout);
}
