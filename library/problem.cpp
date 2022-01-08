// Copyright 2021-2022 Vincent Jacques

#include "problem.hpp"

#include <chrones.hpp>

#include "cuda-utils.hpp"
#include "io.hpp"


namespace ppl {

template<typename Space>
Domain<Space>::Domain(
  Privacy,
  const uint categories_count,
  const uint criteria_count,
  const uint learning_alternatives_count,
  float* learning_alternatives,
  uint* learning_assignments) :
    _categories_count(categories_count),
    _criteria_count(criteria_count),
    _learning_alternatives_count(learning_alternatives_count),
    _learning_alternatives(learning_alternatives),
    _learning_assignments(learning_assignments) {}

template<>
std::shared_ptr<Domain<Host>> Domain<Host>::make(const io::LearningSet& learning_set) {
  CHRONE();

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

  return std::make_shared<Domain<Host>>(
    Privacy(),
    learning_set.categories_count,
    learning_set.criteria_count,
    learning_set.alternatives_count,
    alternatives_,
    assignments_);
}

template<typename Space>
Domain<Space>::~Domain() {
  Space::free(_learning_alternatives);
  Space::free(_learning_assignments);
}

template<typename Space>
DomainView Domain<Space>::get_view() const {
  return {
    _categories_count,
    _criteria_count,
    _learning_alternatives_count,
    MatrixView2D<const float>(_criteria_count, _learning_alternatives_count, _learning_alternatives),
    MatrixView1D<const uint>(_learning_alternatives_count, _learning_assignments),
  };
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
  Privacy,
  std::shared_ptr<Domain<Space>> domain,
  const uint models_count,
  uint* initialization_iteration_indexes,
  float* weights,
  float* profiles) :
    _domain(domain),
    _models_count(models_count),
    _initialization_iteration_indexes(initialization_iteration_indexes),
    _weights(weights),
    _profiles(profiles) {}

template<>
std::shared_ptr<Models<Host>> Models<Host>::make(std::shared_ptr<Domain<Host>> domain, const uint models_count) {
  CHRONE();

  DomainView domain_view = domain->get_view();

  uint* initialization_iteration_indexes = alloc_host<uint>(models_count);
  float* weights = alloc_host<float>(domain_view.criteria_count * models_count);
  float* profiles = alloc_host<float>(domain_view.criteria_count * (domain_view.categories_count - 1) * models_count);

  return std::make_shared<Models>(Privacy(), domain, models_count, initialization_iteration_indexes, weights, profiles);
}

template<>
std::shared_ptr<Models<Host>> Models<Host>::make(
  std::shared_ptr<Domain<Host>> domain,
  const std::vector<io::Model>& models
) {
  CHRONE();

  const uint models_count = models.size();
  auto r = make(domain, models_count);
  auto view = r->get_view();

  for (uint model_index = 0; model_index != models_count; ++model_index) {
    const io::Model& model = models[model_index];
    assert(model.is_valid());

    for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
      view.weights[crit_index][model_index] = model.weights[crit_index];
    }

    for (uint cat_index = 0; cat_index != view.domain.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
        view.profiles[crit_index][cat_index][model_index] = category_profile[crit_index];
      }
    }
  }

  return r;
}

template<>
io::Model Models<Host>::unmake_one(uint model_index) const {
  CHRONE();

  ModelsView view = get_view();

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

  return io::Model(view.domain.criteria_count, view.domain.categories_count, profiles, weights);
}

template<>
std::vector<io::Model> Models<Host>::unmake() const {
  CHRONE();

  ModelsView view = get_view();

  std::vector<io::Model> models;

  models.reserve(view.models_count);
  for (uint model_index = 0; model_index != _models_count; ++model_index) {
    models.push_back(unmake_one(model_index));
  }

  return models;
}

template<typename Space>
Models<Space>::~Models() {
  Space::free(_weights);
  Space::free(_profiles);
}

template<typename Space>
ModelsView Models<Space>::get_view() const {
  DomainView domain = _domain->get_view();
  return {
    domain,
    _models_count,
    MatrixView1D<uint>(_models_count, _initialization_iteration_indexes),
    MatrixView2D<float>(domain.criteria_count, _models_count, _weights),
    MatrixView3D<float>(domain.criteria_count, domain.categories_count - 1, _models_count, _profiles),
  };
}

template class Models<Host>;
template class Models<Device>;

void replicate_models(const Models<Host>& src, Models<Device>* dst) {
  CHRONE();

  DomainView domain = src._domain->get_view();
  copy_host_to_device(
    domain.criteria_count * (domain.categories_count - 1) * src._models_count,
    src._profiles, dst->_profiles);
  copy_host_to_device(
    domain.criteria_count * src._models_count,
    src._weights, dst->_weights);
}

void replicate_profiles(const Models<Device>& src, Models<Host>* dst) {
  CHRONE();

  DomainView domain = src._domain->get_view();
  copy_device_to_host(
    domain.criteria_count * (domain.categories_count - 1) * src._models_count,
    src._profiles, dst->_profiles);
}

}  // namespace ppl
