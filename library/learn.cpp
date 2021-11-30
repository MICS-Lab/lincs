// Copyright 2021 Vincent Jacques

#include "learn.hpp"

#include <vector>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "improve-weights.hpp"


namespace ppl::learn {

io::Model learn_from(const RandomSource& random, const io::LearningSet& learning_set) {
  auto domain = ppl::Domain<Host>::make(learning_set);
  auto start_model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 0, learning_set.categories_count);
  // @todo Improve a population of more than 1 model
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, start_model));

  for (int i = 0; i != 10; ++i) {
    improve_weights::improve_weights(&models);
    std::cerr << "After improve_weights n°" << i << ": "
      << get_accuracy(models, 0) << "/" << learning_set.alternatives_count << std::endl;
    improve_profiles::improve_profiles(random, &models);
    std::cerr << "After improve_profiles n°" << i << ": "
      << get_accuracy(models, 0) << "/" << learning_set.alternatives_count << std::endl;
  }

  return models.unmake().front();
}

}  // namespace ppl::learn
