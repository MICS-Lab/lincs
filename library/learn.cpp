// Copyright 2021 Vincent Jacques

#include "learn.hpp"

#include <vector>
#include <algorithm>

#include "assign.hpp"
#include "improve-profiles.hpp"
#include "improve-weights.hpp"
#include "median-and-max.hpp"


namespace ppl::learn {

// This file make a lot of calls to `get_accuracy`. @todo Consider caching the results of `get_accuracy`

std::vector<uint> partition_models_by_accuracy(ppl::Models<Host>* models) {
  std::vector<uint> model_indexes(models->get_view().models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [models](uint left_model_index, uint right_model_index) {
      return get_accuracy(*models, left_model_index) < get_accuracy(*models, right_model_index);
    });
  return model_indexes;
}

io::Model learn_from(const RandomSource& random, const io::LearningSet& learning_set) {
  auto domain = ppl::Domain<Host>::make(learning_set);
  auto start_model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 0, learning_set.categories_count);
  const uint models_count = 15;
  auto models = ppl::Models<Host>::make(domain, std::vector<ppl::io::Model>(models_count, start_model));

  std::vector<uint> accuracies(models_count, 0);
  auto model_indexes = partition_models_by_accuracy(&models);

  for (int i = 0; i != 3 && accuracies[model_indexes.back()] != learning_set.alternatives_count; ++i) {
    improve_weights::improve_weights(&models);
    improve_profiles::improve_profiles(random, &models);
    model_indexes = partition_models_by_accuracy(&models);
    // @todo Reinitialize models at indexes in lower half of `model_indexes`
    for (uint model_index = 0; model_index != models_count; ++model_index) {
      accuracies[model_index] = get_accuracy(models, model_index);
    }
    std::cerr << "After iteration n°" << i << ": " << std::endl;
    for (uint model_index : model_indexes) {
      std::cerr << " - model " << model_index << ": "
        << accuracies[model_index] << "/" << learning_set.alternatives_count << " " << std::endl;
    }
  }

  return models.unmake_one(model_indexes.back());
}

}  // namespace ppl::learn
