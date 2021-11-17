// Copyright 2021 Vincent Jacques

#include <iostream>
#include <sstream>
#include <fstream>

#include "../library/improve-profiles.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " LEARNING_SET.txt\n"
    << std::endl;
  exit(1);
}

int main(int argc, char* argv[]) {
  if (argc != 2) usage(argv[0]);

  std::ifstream learning_set_file(argv[1]);
  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);

  auto model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 2., learning_set.categories_count);

  auto domain = ppl::improve_profiles::Domain<Host>::make(learning_set);
  auto models = ppl::improve_profiles::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, model));

  const float accuracy_before = ppl::improve_profiles::get_accuracy(models, 0);
  std::cout << "Accuracy before: " << accuracy_before << "/" << learning_set.alternatives_count << std::endl;

  float accuracy_after;
  for (int i = 1; i <= 10; ++i) {
    ppl::improve_profiles::improve_profiles(&models);
    accuracy_after = ppl::improve_profiles::get_accuracy(models, 0);
    std::cout << "Accuracy after iteration n°" << i << ": "
              << accuracy_after << "/" << learning_set.alternatives_count << std::endl;
    if (accuracy_after == learning_set.alternatives_count) {
      break;
    }
  }

  return accuracy_after > accuracy_before ? 0 : 1;
}
