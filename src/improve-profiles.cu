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
  assert(learning_alternatives.s1() == criteria_count);
  assert(learning_alternatives.s0() == learning_alternatives_count);
  assert(learning_assignments.s0() == learning_alternatives_count);
}

template<typename Space>
Domain<Space> Domain<Space>::make(const std::vector<LearningAlternative>& learning_alternatives) {
  assert(!learning_alternatives.empty());
  const int criteria_count = learning_alternatives.front().criteria.size();
  const int alternatives_count = learning_alternatives.size();

  Matrix2D<Host, float> alternatives(criteria_count, alternatives_count);
  Matrix1D<Host, int> assignments(alternatives_count);
  int categories_count = 0;

  for (int alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    const LearningAlternative& alt = learning_alternatives[alt_index];

    categories_count = std::max(categories_count, alt.assignment + 1);

    assert(alt.criteria.size() == criteria_count);
    for (int crit_index = 0; crit_index != criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria[crit_index];
    }
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
      Matrix1D<Space, float>&& thresholds_,
      Matrix3D<Space, float>&& profiles_) :
    domain(domain_),
    models_count(models_count_),
    weights(std::move(weights_)),
    thresholds(std::move(thresholds_)),
    profiles(std::move(profiles_)) {
  assert(weights.s1() == domain.criteria_count);
  assert(weights.s0() == models_count);
  assert(thresholds.s0() == models_count);
  assert(profiles.s2() == domain.criteria_count);
  assert(profiles.s1() == domain.categories_count - 1);
  assert(profiles.s0() == models_count);
}

template<typename Space>
Models<Space> Models<Space>::make(const Domain<Space>& domain, const std::vector<Model>& models) {
  const int models_count = models.size();
  Matrix2D<Host, float> weights(domain.criteria_count, models_count);
  Matrix1D<Host, float> thresholds(models_count);
  Matrix3D<Host, float> profiles(domain.criteria_count, domain.categories_count - 1, models_count);

  for (int model_index = 0; model_index != models_count; ++model_index) {
    const Model& model = models[model_index];

    thresholds[model_index] = model.threshold;

    assert(model.weights.size() == domain.criteria_count);
    for (int crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
      weights[crit_index][model_index] = model.weights[crit_index];
    }

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
    transfer_to<Space>(std::move(thresholds)),
    transfer_to<Space>(std::move(profiles)));
}

template class Models<Host>;
template class Models<Device>;
