// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <algorithm>
#include <utility>
#include <cassert>

#include "cuda-utils.hpp"


template<typename Space>
Domain<Space>::Domain(
      int categories_count_,
      int criteria_count_,
      int learning_alternatives_count_,
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
Domain<Space> Domain<Space>::make(int categories_count, const std::vector<LearningAlternative>& learning_alternatives) {
  assert(categories_count > 1);
  assert(!learning_alternatives.empty());
  const int criteria_count = learning_alternatives.front().criteria.size();
  const int alternatives_count = learning_alternatives.size();

  Matrix2D<Host, float> alternatives(criteria_count, alternatives_count);
  Matrix1D<Host, int> assignments(alternatives_count);

  for (int alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    const LearningAlternative& alt = learning_alternatives[alt_index];

    assert(alt.criteria.size() == criteria_count);
    for (int crit_index = 0; crit_index != criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria[crit_index];
    }

    assert(alt.assignment >= 0 && alt.assignment < categories_count);
    assignments[alt_index] = alt.assignment;
  }

  return Domain(
    categories_count,
    criteria_count,
    alternatives_count,
    transfer_to<Space>(std::move(alternatives)),
    transfer_to<Space>(std::move(assignments)));
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
      const Domain<Space>& domain_,
      int models_count_,
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
Models<Space> Models<Space>::make(const Domain<Space>& domain, const std::vector<Model>& models) {
  const int models_count = models.size();
  Matrix2D<Host, float> weights(domain.criteria_count, models_count);
  Matrix3D<Host, float> profiles(domain.criteria_count, domain.categories_count - 1, models_count);

  for (int model_index = 0; model_index != models_count; ++model_index) {
    const Model& model = models[model_index];

    assert(model.weights.size() == domain.criteria_count);
    for (int crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
      weights[crit_index][model_index] = model.weights[crit_index];
    }

    // @todo Add assertions checking profiles are ordered on each criterion
    assert(model.profiles.size() == domain.categories_count - 1);
    for (int cat_index = 0; cat_index != domain.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      assert(category_profile.size() == domain.criteria_count);
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
int get_assignment(const Models<Space>& models, int model_index, int alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index >= 0 && model_index < models.models_count);
  assert(alternative_index >= 0 && alternative_index < models.domain.learning_alternatives_count);

  for (int category_index = 1; category_index != models.domain.categories_count; ++category_index) {
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
int get_accuracy(const Models<Space>& models, int model_index) {
  // @todo Evaluate if it's worth storing and updating the models' accuracy
  // (instead of recomputing it here)
  int accuracy = 0;
  for (int alternative_index = 0; alternative_index != models.domain.learning_alternatives_count; ++alternative_index) {
    const int expected_assignment = models.domain.learning_assignments[alternative_index];
    const int actual_assignment = get_assignment(models, model_index, alternative_index);
    if (actual_assignment == expected_assignment) {
      ++accuracy;
    }
  }
  return accuracy;
}

template int get_accuracy(const Models<Host>&, int);
