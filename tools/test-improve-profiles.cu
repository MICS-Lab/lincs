// Copyright 2021 Vincent Jacques

#include <iostream>
#include <sstream>
#include <fstream>

#include "../library/improve-profiles.hpp"
#include "../library/stopwatch.hpp"


void usage(char* name) {
  std::cerr <<
    "Usage: " << name << " LEARNING_SET.txt SEED [--device]\n"
    << std::endl;
  exit(1);
}

template<typename Space>
bool loop(
    const RandomSource& random,
    const ppl::io::LearningSet& learning_set,
    ppl::improve_profiles::Models<Space>* models
) {
  STOPWATCH("test-improve-profiles loop");

  const float accuracy_before = ppl::improve_profiles::get_accuracy(*models, 0);
  std::cout << "Accuracy before: " << accuracy_before << "/" << learning_set.alternatives_count << std::endl;

  float accuracy_after = 0;
  for (unsigned int i = 1; i <= 10; ++i) {
    STOPWATCH("test-improve-profiles loop iteration");
    ppl::improve_profiles::improve_profiles(random, models);
    accuracy_after = ppl::improve_profiles::get_accuracy(*models, 0);
    std::cout << "Accuracy after iteration n°" << i << ": "
              << accuracy_after << "/" << learning_set.alternatives_count << std::endl;
    if (accuracy_after == learning_set.alternatives_count) {
      break;
    }
  }

  return accuracy_after > accuracy_before;
}

int main(int argc, char* argv[]) {
  STOPWATCH("test-improve-profiles");

  if (argc < 3) usage(argv[0]);

  std::ifstream learning_set_file(argv[1]);

  unsigned int seed;
  try {
    seed = std::stoi(argv[2]);
  } catch (std::invalid_argument) {
    usage(argv[0]);
  }

  const bool run_on_device = argc > 3 && std::string(argv[3]) == "--device";

  auto learning_set = ppl::io::LearningSet::load_from(learning_set_file);

  auto model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 2., learning_set.categories_count);

  auto domain = ppl::improve_profiles::Domain<Host>::make(learning_set);
  auto models = ppl::improve_profiles::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, model));

  RandomSource random;

  if (run_on_device) {
    auto device_domain = domain.clone_to<Device>();
    auto device_models = models.clone_to<Device>(device_domain);
    random.init_for_device(seed);
    return loop(random, learning_set, &device_models) ? 0 : 1;
  } else {
    random.init_for_host(seed);
    return loop(random, learning_set, &models) ? 0 : 1;
  }
}
