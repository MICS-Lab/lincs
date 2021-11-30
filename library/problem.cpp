// Copyright 2021 Vincent Jacques

#include "problem.hpp"

#include "cuda-utils.hpp"
#include "io.hpp"


namespace ppl {

template<typename Space>
Domain<Space>::Domain(
  const uint categories_count_,
  const uint criteria_count_,
  const uint learning_alternatives_count_,
  float* learning_alternatives_,
  uint* learning_assignments_) :
    categories_count(categories_count_),
    criteria_count(criteria_count_),
    learning_alternatives_count(learning_alternatives_count_),
    learning_alternatives(learning_alternatives_),
    learning_assignments(learning_assignments_) {}

template<>
Domain<Host> Domain<Host>::make(const io::LearningSet& learning_set) {
  assert(learning_set.is_valid());

  float* alternatives_ = alloc_host<float>(learning_set.criteria_count * learning_set.alternatives_count);
  MatrixView2D<float> alternatives(learning_set.criteria_count, learning_set.alternatives_count, alternatives_);
  uint* assignments_ = alloc_host<uint>(learning_set.alternatives_count);
  MatrixView1D<uint> assignments(learning_set.alternatives_count, assignments_);

  for (uint alt_index = 0; alt_index != learning_set.alternatives_count; ++alt_index) {
    const io::ClassifiedAlternative& alt = learning_set.alternatives[alt_index];

    for (uint crit_index = 0; crit_index != learning_set.criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria_values[crit_index];
    }

    assignments[alt_index] = alt.assigned_category;
  }

  return Domain(
    learning_set.categories_count,
    learning_set.criteria_count,
    learning_set.alternatives_count,
    alternatives_,
    assignments_);
}

template<typename Space>
Domain<Space>::~Domain() {
  Space::free(learning_alternatives);
  Space::free(learning_assignments);
}

template<typename Space>
DomainView Domain<Space>::get_view() const {
  return {
    categories_count,
    criteria_count,
    learning_alternatives_count,
    MatrixView2D<const float>(criteria_count, learning_alternatives_count, learning_alternatives),
    MatrixView1D<const uint>(learning_alternatives_count, learning_assignments),
  };
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
  const Domain<Space>& domain_,
  const uint models_count_,
  float* weights_,
  float* profiles_) :
    domain(domain_),
    models_count(models_count_),
    weights(weights_),
    profiles(profiles_) {}

template<>
Models<Host> Models<Host>::make(const Domain<Host>& domain, const std::vector<io::Model>& models) {
  DomainView domain_view = domain.get_view();
  const uint models_count = models.size();
  float* weights_ = alloc_host<float>(domain_view.criteria_count * models_count);
  MatrixView2D<float> weights(domain_view.criteria_count, models_count, weights_);
  float* profiles_ = alloc_host<float>(domain_view.criteria_count * (domain_view.categories_count - 1) * models_count);
  MatrixView3D<float> profiles(domain_view.criteria_count, domain_view.categories_count - 1, models_count, profiles_);

  for (uint model_index = 0; model_index != models_count; ++model_index) {
    const io::Model& model = models[model_index];
    assert(model.is_valid());

    for (uint crit_index = 0; crit_index != domain_view.criteria_count; ++crit_index) {
      weights[crit_index][model_index] = model.weights[crit_index];
    }

    for (uint cat_index = 0; cat_index != domain_view.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      for (uint crit_index = 0; crit_index != domain_view.criteria_count; ++crit_index) {
        profiles[crit_index][cat_index][model_index] = category_profile[crit_index];
      }
    }
  }

  return Models(domain, models_count, weights_, profiles_);
}

template<>
std::vector<io::Model> Models<Host>::unmake() const {
  ModelsView view = get_view();

  std::vector<io::Model> models;
  models.reserve(view.models_count);
  for (uint model_index = 0; model_index != models_count; ++model_index) {
    std::vector<std::vector<float>> profiles(view.domain.categories_count - 1);
    for (uint cat_index = 0; cat_index != view.domain.categories_count - 1; ++cat_index) {
      profiles[cat_index].reserve(view.domain.criteria_count);
      for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
        profiles[cat_index].push_back(view.profiles[crit_index][cat_index][model_index]);
      }
    }

    std::vector<float> weights;
    weights.reserve(view.domain.criteria_count);
    for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
      weights.push_back(view.weights[crit_index][model_index]);
    }

    models.push_back(io::Model(view.domain.criteria_count, view.domain.categories_count, profiles, weights));
  }
  return models;
}

template<typename Space>
Models<Space>::~Models() {
  Space::free(weights);
  Space::free(profiles);
}

template<typename Space>
ModelsView Models<Space>::get_view() const {
  DomainView domain_view = domain.get_view();
  return {
    domain_view,
    models_count,
    MatrixView2D<float>(domain_view.criteria_count, models_count, weights),
    MatrixView3D<float>(domain_view.criteria_count, domain_view.categories_count - 1, models_count, profiles),
  };
}

template class Models<Host>;
template class Models<Device>;

}  // namespace ppl
