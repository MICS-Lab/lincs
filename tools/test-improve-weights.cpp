// Copyright 2021 Vincent Jacques

#include <iostream>
#include <sstream>
#include <fstream>

#include "../library/assign.hpp"
#include "../library/improve-weights.hpp"
#include "../library/stopwatch.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " LEARNING_SET.txt MODEL.txt\n"
    << std::endl;
  exit(1);
}

int main(int argc, char* argv[]) {
  STOPWATCH("test-improve-weights");

  if (argc != 3) usage(argv[0]);

  std::ifstream learning_set_file(argv[1]);
  std::ifstream model_file(argv[2]);

  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);
  auto model = ppl::io::Model::load_from(model_file);

  auto domain = ppl::Domain<Host>::make(learning_set);
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, model));

  uint accuracy = ppl::get_accuracy(models, 0);
  std::cout << "Accuracy before: " << accuracy << "/" << learning_set.alternatives_count << std::endl;

  ppl::improve_weights(&models);

  accuracy = ppl::get_accuracy(models, 0);
  std::cout << "Accuracy after: " << accuracy << "/" << learning_set.alternatives_count << std::endl;

  return accuracy == learning_set.alternatives_count ? 0 : 1;
}
