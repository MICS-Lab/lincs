// Copyright 2023 Vincent Jacques

#include "weights-profiles-breed-mrsort.hpp"

#include <map>

#include "exception.hpp"
#include "../median-and-max.hpp"


namespace lincs {

WeightsProfilesBreedMrSortLearning::Models WeightsProfilesBreedMrSortLearning::Models::make(const Problem& problem, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed) {
  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.categories.size();
  const unsigned alternatives_count = learning_set.alternatives.size();

  Array2D<Host, float> alternatives(criteria_count, alternatives_count, uninitialized);
  Array1D<Host, unsigned> assignments(alternatives_count, uninitialized);

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const Alternative& alt = learning_set.alternatives[alternative_index];

    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      alternatives[criterion_index][alternative_index] = alt.profile[criterion_index];
    }

    assignments[alternative_index] = *alt.category_index;
  }

  std::vector<unsigned> model_indexes(models_count);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);

  Array2D<Host, float> weights(criteria_count, models_count, uninitialized);
  Array3D<Host, float> profiles(criteria_count, (categories_count - 1), models_count, uninitialized);
  Array1D<Host, unsigned> accuracies(models_count, zeroed);

  std::vector<std::mt19937> urbgs(models_count);
  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    urbgs[model_index].seed(random_seed * (model_index + 1));
  }

  return {
    problem,
    categories_count,
    criteria_count,
    alternatives_count,
    std::move(alternatives),
    std::move(assignments),
    0,
    models_count,
    std::move(model_indexes),
    std::move(weights),
    std::move(profiles),
    std::move(accuracies),
    std::move(urbgs),
  };
}

Model WeightsProfilesBreedMrSortLearning::Models::get_model(const unsigned model_index) const {
  assert(model_index < models_count);

  std::vector<float> model_weights;
  model_weights.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model_weights.push_back(weights[criterion_index][model_index]);
  }
  Model::SufficientCoalitions coalitions{Model::SufficientCoalitions::weights, model_weights};

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(categories_count - 1);
  for (unsigned cat_index = 0; cat_index != categories_count - 1; ++cat_index) {
    std::vector<float> boundary_profile;
    boundary_profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      boundary_profile.push_back(profiles[criterion_index][cat_index][model_index]);
    }
    boundaries.emplace_back(boundary_profile, coalitions);
  }

  return Model{problem, boundaries};
}

Model WeightsProfilesBreedMrSortLearning::perform() {
  profiles_initialization_strategy.initialize_profiles(models.model_indexes.begin(), models.model_indexes.end());

  unsigned iterations_without_progress = 0;
  // Limit is arbitrary; unit tests show 40 is required, so 100 seems OK with some margin
  while (iterations_without_progress < 100) {
    const int previous_best_accuracy = models.get_best_accuracy();

    // Improve
    // @todo Shouldn't we swap the next two lines? (To have optimal weights before breeding)
    weights_optimization_strategy.optimize_weights();
    profiles_improvement_strategy.improve_profiles();

    // Sort model_indexes by increasing model accuracy
    for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
      models.accuracies[model_index] = compute_accuracy(model_index);
    }
    std::sort(
      models.model_indexes.begin(), models.model_indexes.end(),
      [this](unsigned left_model_index, unsigned right_model_index) {
        return models.accuracies[left_model_index] < models.accuracies[right_model_index];
      }
    );

    // Interrupt if no progress
    const int new_best_accuracy = models.get_best_accuracy();
    if (new_best_accuracy > previous_best_accuracy) {
      iterations_without_progress = 0;
    } else {
      ++iterations_without_progress;
    }

    // Succeed?
    // @todo Let termination strategy access the models; remove its parameters
    if (termination_strategy.terminate(models.iteration_index, models.get_best_accuracy())) {
      return models.get_model(models.model_indexes.back());
    }

    // Breed
    // @todo Introduce breeding extension point
    profiles_initialization_strategy.initialize_profiles(models.model_indexes.begin(), models.model_indexes.begin() + models.models_count / 2);

    ++models.iteration_index;
  }

  // Fail
  throw LearningFailureException();
}

unsigned WeightsProfilesBreedMrSortLearning::compute_accuracy(const unsigned model_index) {
  unsigned accuracy = 0;

  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    if (is_correctly_assigned(model_index, alternative_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

bool WeightsProfilesBreedMrSortLearning::is_correctly_assigned(
    const unsigned model_index,
    const unsigned alternative_index) {
  const unsigned expected_assignment = models.learning_assignments[alternative_index];
  const unsigned actual_assignment = get_assignment(models, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

unsigned WeightsProfilesBreedMrSortLearning::get_assignment(const Models& models, const unsigned model_index, const unsigned alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index < models.models_count);
  assert(alternative_index < models.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = models.categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
      const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
      const float profile_value = models.profiles[criterion_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[criterion_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

}  // namespace lincs
