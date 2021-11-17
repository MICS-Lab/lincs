// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <algorithm>
#include <utility>
#include <cassert>
#include <random>

#include "cuda-utils.hpp"


namespace ppl::improve_profiles {

template<typename Space>
Domain<Space>::Domain(
      const int categories_count_,
      const int criteria_count_,
      const int learning_alternatives_count_,
      Matrix2D<Space, float>&& learning_alternatives_,
      Matrix1D<Space, int>&& learning_assignments_) :
    categories_count(categories_count_),
    criteria_count(criteria_count_),
    learning_alternatives_count(learning_alternatives_count_),
    learning_alternatives(std::move(learning_alternatives_)),
    learning_assignments(std::move(learning_assignments_)) {
  assert(categories_count > 1);
  assert(criteria_count > 0);
  assert(learning_alternatives_count > 0);
  assert(learning_alternatives.s1() == criteria_count);
  assert(learning_alternatives.s0() == learning_alternatives_count);
  assert(learning_assignments.s0() == learning_alternatives_count);
}

template<typename Space>
Domain<Space> Domain<Space>::make(const io::LearningSet& learning_set) {
  assert(learning_set.is_valid());

  Matrix2D<Host, float> alternatives(learning_set.criteria_count, learning_set.alternatives_count);
  Matrix1D<Host, int> assignments(learning_set.alternatives_count);

  for (int alt_index = 0; alt_index != learning_set.alternatives_count; ++alt_index) {
    const io::ClassifiedAlternative& alt = learning_set.alternatives[alt_index];

    for (int crit_index = 0; crit_index != learning_set.criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria_values[crit_index];
    }

    assignments[alt_index] = alt.assigned_category;
  }

  return Domain(
    learning_set.categories_count,
    learning_set.criteria_count,
    learning_set.alternatives_count,
    transfer_to<Space>(std::move(alternatives)),
    transfer_to<Space>(std::move(assignments)));
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
      const Domain<Space>& domain_,
      const int models_count_,
      Matrix2D<Space, float>&& weights_,
      Matrix3D<Space, float>&& profiles_) :
    domain(domain_),
    models_count(models_count_),
    weights(std::move(weights_)),
    profiles(std::move(profiles_)) {
  assert(weights.s1() == domain.criteria_count);
  assert(weights.s0() == models_count);
  assert(profiles.s2() == domain.criteria_count);
  assert(profiles.s1() == domain.categories_count - 1);
  assert(profiles.s0() == models_count);
}

template<typename Space>
Models<Space> Models<Space>::make(const Domain<Space>& domain, const std::vector<io::Model>& models) {
  const int models_count = models.size();
  Matrix2D<Host, float> weights(domain.criteria_count, models_count);
  Matrix3D<Host, float> profiles(domain.criteria_count, domain.categories_count - 1, models_count);

  for (int model_index = 0; model_index != models_count; ++model_index) {
    const io::Model& model = models[model_index];
    assert(model.is_valid());

    for (int crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
      weights[crit_index][model_index] = model.weights[crit_index];
    }

    for (int cat_index = 0; cat_index != domain.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      for (int crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
        profiles[crit_index][cat_index][model_index] = category_profile[crit_index];
      }
    }
  }

  return Models(
    domain,
    models_count,
    transfer_to<Space>(std::move(weights)),
    transfer_to<Space>(std::move(profiles)));
}

template class Models<Host>;
template class Models<Device>;

template<typename Space>
int get_assignment(const Models<Space>& models, const int model_index, const int alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index >= 0 && model_index < models.models_count);
  assert(alternative_index >= 0 && alternative_index < models.domain.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (int category_index = models.domain.categories_count - 1; category_index != 0; --category_index) {
    const int profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (int crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alternative_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

template<typename Space>
int get_accuracy(const Models<Space>& models, const int model_index) {
  // @todo Evaluate if it's worth storing and updating the models' accuracy
  // (instead of recomputing it here)
  int accuracy = 0;
  for (int alternative_index = 0; alternative_index != models.domain.learning_alternatives_count; ++alternative_index) {
    // Map (embarassingly parallel)
    const int expected_assignment = models.domain.learning_assignments[alternative_index];
    const int actual_assignment = get_assignment(models, model_index, alternative_index);
    // Single-key reduce (atomicAdd)
    if (actual_assignment == expected_assignment) {
      ++accuracy;
    }
  }
  return accuracy;
}

template int get_accuracy(const Models<Host>&, int);

Desirability compute_move_desirability(
  const Models<Host>& models,
  const int model_index,
  const int profile_index,
  const int criterion_index,
  const float destination
) {
  Desirability desirability;

  const float current_position = models.profiles[criterion_index][profile_index][model_index];
  const float weight = models.weights[criterion_index][model_index];

  for (int alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    // Map (embarassigly parallel)
    const float value = models.domain.learning_alternatives[criterion_index][alt_index];
    const int learning_assignment = models.domain.learning_assignments[alt_index];
    const int model_assignment = get_assignment(models, model_index, alt_index);

    // @todo Factorize with get_assignment
    float weight_at_or_above_profile = 0;
    for (int crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }

    // Single-key reduce (atomicAdd)

    // These imbricated conditionals could be factorized, but this form has the benefit
    // of being a direct translation of the top of page 78 of Sobrie's thesis.
    // Correspondance:
    // - learning_assignment: bottom index of A*
    // - model_assignment: top index of A*
    // - profile_index: h
    // - destination: b_j +/- \delta
    // - current_position: b_j
    // - value: a_j
    // - weight_at_or_above_profile: \sigma
    // - weight: w_j
    // - 1: \lambda
    if (destination > current_position) {
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight < 1) ++desirability.v;
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight >= 1) ++desirability.w;
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight < 1) ++desirability.q;
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination > value
        && value >= current_position) ++desirability.r;
      if (
        learning_assignment < profile_index
        && model_assignment > profile_index
        && destination > value
        && value >= current_position) ++desirability.t;
    } else {
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight >= 1) ++desirability.v;
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight < 1) ++desirability.w;
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight >= 1) ++desirability.q;
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination <= value
        && value < current_position) ++desirability.r;
      if (
        learning_assignment > profile_index + 1
        && model_assignment < profile_index + 1
        && destination < value
        && value <= current_position) ++desirability.t;
    }
  }

  return desirability;
}

void improve_model_profile(
  Models<Host>* models,
  const int model_index,
  const int profile_index,
  const int criterion_index
) {
  std::random_device rd;
  std::mt19937 g(rd());

  // WARNING: We're assuming all criteria have values in [0, 1]
  // @todo Can we relax this assumption?
  // This is consistent with our comment in the header file, but slightly less generic than Sobrie's thesis
  const float lowest_destination =
    profile_index == 0 ? 0. :
    models->profiles[criterion_index][profile_index - 1][model_index];
  const float highest_destination =
    profile_index == models->domain.categories_count - 2 ? 1. :
    models->profiles[criterion_index][profile_index + 1][model_index];
  std::uniform_real_distribution<> destination_distribution(lowest_destination, highest_destination);

  float best_destination = models->profiles[criterion_index][profile_index][model_index];
  float best_desirability = Desirability().value();
  // Not sure about this part: we're considering an arbitrary number of possible moves as described in
  // Mousseau's prez-mics-2018(8).pdf, but:
  //  - this is wasteful when there are fewer alternatives in the interval
  //  - this is not strictly consistent with, albeit much simpler than, Sobrie's thesis
  // @todo Ask Vincent Mousseau about the following:
  // We could consider only a finite set of values for b_j described as follows:
  // - sort all the 'a_j's
  // - compute all midpoints between two successive 'a_j'
  // - add two extreme values (0 and 1, or above the greatest a_j and below the smallest a_j)
  // Then instead of taking a random values in destination_distribution, we'd take a random subset of
  // the intersection of these midpoints with that interval.
  for (int n = 0; n < 1024; ++n) {
    // Map (embarassigly parallel)
    const float destination = destination_distribution(g);
    const float desirability = compute_move_desirability(
      *models, model_index, profile_index, criterion_index, destination).value();
    // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  // @todo Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (std::uniform_real_distribution<>(0.f, 1.f)(g) <= best_desirability) {
    models->profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

void improve_model_profile(
  Models<Host>* models,
  const int model_index,
  const int profile_index,
  const std::vector<int>& criterion_indexes
) {
  // Loop is not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (int criterion_index : criterion_indexes) {
    improve_model_profile(models, model_index, profile_index, criterion_index);
  }
}

template<>
void improve_profiles<Host>(Models<Host>* models) {
  std::random_device rd;
  std::mt19937 g(rd());

  std::vector<int> criterion_indexes(models->domain.criteria_count, 0);
  std::iota(criterion_indexes.begin(), criterion_indexes.end(), 0);

  // Outer loop is embarassingly parallel
  for (int model_index = 0; model_index != models->models_count; ++model_index) {
    // Inner loop is not parallel because iteration N+1 relies on side effect in iteration N
    // (We could challenge this aspect of the algorithm described by Sobrie)
    for (int profile_index = 0; profile_index != models->domain.categories_count - 1; ++profile_index) {
      std::shuffle(criterion_indexes.begin(), criterion_indexes.end(), g);
      improve_model_profile(models, model_index, profile_index, criterion_indexes);
    }
  }
}

}  // namespace ppl::improve_profiles
