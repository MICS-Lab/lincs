// Copyright 2023 Vincent Jacques

#include "mrsort-by-weights-profiles-breed.hpp"

#include <map>

#include "exception.hpp"


namespace lincs {

LearnMrsortByWeightsProfilesBreed::LearningData LearnMrsortByWeightsProfilesBreed::LearningData::make(const Problem& problem, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed) {
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

Model LearnMrsortByWeightsProfilesBreed::LearningData::get_model(const unsigned model_index) const {
  assert(model_index < models_count);

  std::vector<float> model_weights;
  model_weights.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model_weights.push_back(weights[criterion_index][model_index]);
  }
  SufficientCoalitions coalitions{SufficientCoalitions::weights, model_weights};

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

Model LearnMrsortByWeightsProfilesBreed::perform() {
  profiles_initialization_strategy.initialize_profiles(0, learning_data.models_count);

  unsigned iterations_without_progress = 0;
  // Limit is arbitrary; unit tests show 40 is required, so 100 seems OK with some margin
  while (iterations_without_progress < 100) {
    const unsigned previous_best_accuracy = learning_data.get_best_accuracy();

    // Improve
    weights_optimization_strategy.optimize_weights();
    profiles_improvement_strategy.improve_profiles();

    // Sort model_indexes by increasing model accuracy
    for (unsigned model_index = 0; model_index != learning_data.models_count; ++model_index) {
      learning_data.accuracies[model_index] = compute_accuracy(model_index);
    }
    std::sort(
      learning_data.model_indexes.begin(), learning_data.model_indexes.end(),
      [this](unsigned left_model_index, unsigned right_model_index) {
        return learning_data.accuracies[left_model_index] < learning_data.accuracies[right_model_index];
      }
    );

    // Interrupt if no progress
    const unsigned new_best_accuracy = learning_data.get_best_accuracy();
    if (new_best_accuracy > previous_best_accuracy) {
      iterations_without_progress = 0;
    } else {
      ++iterations_without_progress;
    }

    // Succeed?
    if (new_best_accuracy == learning_data.learning_alternatives_count || termination_strategy.terminate()) {
      return learning_data.get_model(learning_data.model_indexes.back());
    }

    // Breed
    breeding_strategy.breed();

    // Observe
    for (auto observer : observers) {
      observer->after_iteration();
    }

    ++learning_data.iteration_index;
  }

  // Fail
  throw LearningFailureException();
}

unsigned LearnMrsortByWeightsProfilesBreed::compute_accuracy(const unsigned model_index) {
  unsigned accuracy = 0;

  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    if (is_correctly_assigned(model_index, alternative_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

bool LearnMrsortByWeightsProfilesBreed::is_correctly_assigned(
    const unsigned model_index,
    const unsigned alternative_index) {
  const unsigned expected_assignment = learning_data.learning_assignments[alternative_index];
  const unsigned actual_assignment = get_assignment(learning_data, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

unsigned LearnMrsortByWeightsProfilesBreed::get_assignment(const LearningData& learning_data, const unsigned model_index, const unsigned alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  // Same question in accuracy-heuristic-on-gpu.cu
  assert(model_index < learning_data.models_count);
  assert(alternative_index < learning_data.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      const float alternative_value = learning_data.learning_alternatives[criterion_index][alternative_index];
      const float profile_value = learning_data.profiles[criterion_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += learning_data.weights[criterion_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

}  // namespace lincs
