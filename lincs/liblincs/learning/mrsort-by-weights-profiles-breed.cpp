// Copyright 2023 Vincent Jacques

#include "mrsort-by-weights-profiles-breed.hpp"

#include <map>
#include <numeric>

#include "../chrones.hpp"
#include "../unreachable.hpp"
#include "exception.hpp"


namespace lincs {

LearnMrsortByWeightsProfilesBreed::LearningData::LearningData(
    const Problem& problem_,
    const Alternatives& learning_set,
    const unsigned models_count_,
    const unsigned random_seed
) :
  PreProcessedLearningSet(problem_, learning_set),
  iteration_index(0),
  models_count(models_count_),
  model_indexes(models_count),
  weights(criteria_count, models_count, uninitialized),
  profile_ranks(criteria_count, boundaries_count, models_count, uninitialized),
  accuracies(models_count, zeroed),
  urbgs(models_count)
{
  CHRONE();

  std::iota(model_indexes.begin(), model_indexes.end(), 0);

  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    urbgs[model_index].seed(random_seed * (model_index + 1));
  }
}

Model LearnMrsortByWeightsProfilesBreed::LearningData::get_model(const unsigned model_index) const {
  CHRONE();

  assert(model_index < models_count);

  std::vector<float> model_weights;
  model_weights.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model_weights.push_back(weights[criterion_index][model_index]);
  }
  SufficientCoalitions coalitions{SufficientCoalitions::weights, model_weights};

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<float> boundary_profile;
    boundary_profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const unsigned profile_rank = profile_ranks[criterion_index][boundary_index][model_index];
      const float profile_value = sorted_values[criterion_index][profile_rank];
      boundary_profile.push_back(profile_value);
    }
    boundaries.emplace_back(boundary_profile, coalitions);
  }

  return Model{problem, boundaries};
}

Model LearnMrsortByWeightsProfilesBreed::perform() {
  CHRONE();

  profiles_initialization_strategy.initialize_profiles(0, learning_data.models_count);

  while (true) {
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

    // Succeed?
    if (learning_data.get_best_accuracy() == learning_data.alternatives_count || termination_strategy.terminate()) {
      for (auto observer : observers) {
        observer->before_return();
      }
      return learning_data.get_model(learning_data.model_indexes.back());
    }

    // Breed
    // @todo(Feature, later) Keep the best model and reinit half of the others
    breeding_strategy.breed();

    // Observe
    for (auto observer : observers) {
      observer->after_iteration();
    }

    ++learning_data.iteration_index;
  }

  unreachable();
}

unsigned LearnMrsortByWeightsProfilesBreed::compute_accuracy(const unsigned model_index) {
  unsigned accuracy = 0;

  for (unsigned alternative_index = 0; alternative_index != learning_data.alternatives_count; ++alternative_index) {
    if (is_correctly_assigned(model_index, alternative_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

bool LearnMrsortByWeightsProfilesBreed::is_correctly_assigned(
    const unsigned model_index,
    const unsigned alternative_index) {
  const unsigned expected_assignment = learning_data.assignments[alternative_index];
  const unsigned actual_assignment = get_assignment(learning_data, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

unsigned LearnMrsortByWeightsProfilesBreed::get_assignment(const LearningData& learning_data, const unsigned model_index, const unsigned alternative_index) {
  // @todo(Performance, later) Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  // Same question in accuracy-heuristic-on-gpu.cu
  assert(model_index < learning_data.models_count);
  assert(alternative_index < learning_data.alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_better_than_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      const unsigned alternative_rank = learning_data.performance_ranks[criterion_index][alternative_index];
      const unsigned profile_rank = learning_data.profile_ranks[criterion_index][profile_index][model_index];
      const bool is_better = alternative_rank >= profile_rank;
      if (is_better) {
        weight_at_or_better_than_profile += learning_data.weights[criterion_index][model_index];
      }
    }
    if (weight_at_or_better_than_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

}  // namespace lincs
