// Copyright 2023-2024 Vincent Jacques

#include "mrsort-by-weights-profiles-breed.hpp"

#include <iostream>
#include <map>
#include <numeric>

#include "../chrones.hpp"
#include "../unreachable.hpp"
#include "exception.hpp"


namespace lincs {

LearnMrsortByWeightsProfilesBreed::LearningData::LearningData(
    const Problem& problem,
    const Alternatives& learning_set,
    const unsigned models_count_,
    const unsigned random_seed
) :
  PreProcessedLearningSet(problem, learning_set),
  models_count(models_count_),
  random_generators(models_count),
  iteration_index(0),
  model_indexes(models_count),
  accuracies(models_count, zeroed),
  low_profile_ranks(models_count, boundaries_count, criteria_count, uninitialized),
  high_profile_rank_indexes(criteria_count, uninitialized),
  high_profile_ranks(models_count, boundaries_count, count_single_peaked_criteria(), uninitialized),
  weights(models_count, criteria_count, uninitialized)
{
  CHRONE();

  std::iota(model_indexes.begin(), model_indexes.end(), 0);

  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    random_generators[model_index].seed(random_seed * (model_index + 1));
  }

  unsigned count = 0;
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    if (single_peaked[criterion_index]) {
      high_profile_rank_indexes[criterion_index] = count;
      ++count;
    }
  }
  assert(high_profile_ranks.s0() == count);
}

unsigned LearnMrsortByWeightsProfilesBreed::LearningData::count_single_peaked_criteria() const {
  unsigned count = 0;
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    if (single_peaked[criterion_index]) {
      ++count;
    }
  }
  return count;
}

Model LearnMrsortByWeightsProfilesBreed::LearningData::get_model(const unsigned model_index) const {
  CHRONE();

  assert(model_index < models_count);

  std::vector<float> model_weights;
  model_weights.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model_weights.push_back(weights[model_index][criterion_index]);
  }
  SufficientCoalitions coalitions = SufficientCoalitions(SufficientCoalitions::Weights(model_weights));

  std::vector<PreProcessedBoundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>> boundary_profile;
    boundary_profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const unsigned low_profile_rank = low_profile_ranks[model_index][boundary_index][criterion_index];
      if (single_peaked[criterion_index]) {
        const unsigned high_profile_rank = high_profile_ranks[model_index][boundary_index][high_profile_rank_indexes[criterion_index]];
        boundary_profile.push_back(std::make_pair(low_profile_rank, high_profile_rank));
      } else {
        boundary_profile.push_back(low_profile_rank);
      }
    }
    boundaries.emplace_back(boundary_profile, coalitions);
  }

  return post_process(boundaries);
}

#ifndef NDEBUG
bool LearnMrsortByWeightsProfilesBreed::LearningData::model_is_correct(const unsigned model_index) const {
  try {
    get_model(model_index);
  } catch (const DataValidationException& e) {
    std::cerr << "Model " << model_index << " is incorrect: " << e.what() << std::endl;
    return false;
  }
  return true;
}

bool LearnMrsortByWeightsProfilesBreed::LearningData::models_are_correct() const {
  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    if (!model_is_correct(model_index)) {
      return false;
    }
  }
  return true;
}
#endif


Model LearnMrsortByWeightsProfilesBreed::perform() {
  CHRONE();

  profiles_initialization_strategy.initialize_profiles(0, learning_data.models_count);

  assert(learning_data.models_are_correct());

  while (true) {
    // Improve
    weights_optimization_strategy.optimize_weights(0, learning_data.models_count);
    profiles_improvement_strategy.improve_profiles(0, learning_data.models_count);

    assert(learning_data.models_are_correct());

    // @todo(Feature, later) Rework this main loop. Its current problems:
    //   - we return models that have gone through a last profiles improvement, but their weights have not been optimized
    //   - we decide to stop the learning based on the accuracy of those models in this weird state
    // Beware, if optimize_weights is run after improve_profiles, it must also be run during the breeding strategy.

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

bool LearnMrsortByWeightsProfilesBreed::is_accepted(
  const LearnMrsortByWeightsProfilesBreed::LearningData& learning_data,
  const unsigned model_index,
  const unsigned boundary_index,
  const unsigned criterion_index,
  const unsigned alternative_index
) {
  const unsigned alternative_rank = learning_data.performance_ranks[criterion_index][alternative_index];
  const unsigned low_profile_rank = learning_data.low_profile_ranks[model_index][boundary_index][criterion_index];
  if (learning_data.single_peaked[criterion_index]) {
    const unsigned high_profile_rank = learning_data.high_profile_ranks[model_index][boundary_index][learning_data.high_profile_rank_indexes[criterion_index]];
    return low_profile_rank <= alternative_rank && alternative_rank <= high_profile_rank;
  } else {
    return low_profile_rank <= alternative_rank;
  }
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
    const unsigned boundary_index = category_index - 1;
    float accepted_weight = 0;
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      if (is_accepted(learning_data, model_index, boundary_index, criterion_index, alternative_index)) {
        accepted_weight += learning_data.weights[model_index][criterion_index];
      }
    }
    if (accepted_weight >= 1) {
      return category_index;
    }
  }
  return 0;
}

}  // namespace lincs
