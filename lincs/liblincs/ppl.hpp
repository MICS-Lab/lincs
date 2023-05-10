#line 1 "uint.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef UINT_HPP_
#define UINT_HPP_

typedef unsigned int uint;

#endif  // UINT_HPP_
#line 1 "io.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef IO_HPP_
#define IO_HPP_

#include <iostream>
#include <optional>
#include <string>
#include <vector>


namespace ppl::io {

// Classes for easy input/output of domain objects

struct Model {
  Model(
    uint criteria_count,
    uint categories_count,
    const std::vector<std::vector<float>>& profiles,
    // @todo Add min/max values for each criterion.
    // Then remove (everywhere) the assumption that these values are 0 and 1.
    const std::vector<float>& weights);

  std::optional<std::string> validate() const;
  bool is_valid() const { return !validate(); }

  void save_to(std::ostream&) const;
  static Model load_from(std::istream&);
  static Model make_homogeneous(uint criteria_count, float weights_sum, uint categories_count);

  const uint criteria_count;
  const uint categories_count;
  std::vector<std::vector<float>> profiles;
  std::vector<float> weights;
};

struct ClassifiedAlternative {
  ClassifiedAlternative(
    const std::vector<float>& criteria_values,
    uint assigned_category);

  const std::vector<float> criteria_values;
  uint assigned_category;  // @todo Fix this inconsistency: everything but that is const in this module
};

struct LearningSet {
  LearningSet(
    uint criteria_count,
    uint categories_count,
    uint alternatives_count,
    const std::vector<ClassifiedAlternative>& alternatives);

  std::optional<std::string> validate() const;
  bool is_valid() const { return !validate(); }

  void save_to(std::ostream&) const;
  static LearningSet load_from(std::istream&);

  const uint criteria_count;
  const uint categories_count;
  const uint alternatives_count;
  std::vector<ClassifiedAlternative> alternatives;
};

}  // namespace ppl::io

#endif  // IO_HPP_
#line 1 "problem.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include <memory>
#include <vector>

#include <lov-e.hpp>

// Commented during copy-paste from PPL: #include "io.hpp"
// Commented during copy-paste from PPL: #include "uint.hpp"


namespace ppl {

/*
The constants of the problem, i.e. the sizes of the domain, and the learning set.

@todo Split Domain and DomainView class into Domain proper (sizes, labels, etc.) and LearningSet (classified alternatives)
*/
template<typename Space>
struct DomainView {
  const uint categories_count;
  const uint criteria_count;
  const uint learning_alternatives_count;

  ArrayView2D<Space, const float> learning_alternatives;
  // First index: index of criterion, from `0` to `criteria_count - 1`
  // Second index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Values are pre-normalized on each criterion so that the possible values are from `0.0` to `1.0`.
  // @todo Can we relax this assumption?
  //  - going from `-infinity` to `+infinity` might be possible
  //  - or we can extract the smallest and greatest value of each criterion on all the alternatives
  //  - to handle criterion where a lower value is better, we'd need to store an additional boolean indicator

  ArrayView1D<Space, const uint> learning_assignments;
  // Index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // Possible values: from `0` to `categories_count - 1`

  template<typename S = Space, typename = std::enable_if_t<!std::is_same_v<S, Anywhere>>>
  operator const DomainView<Anywhere>&() const {
    return *reinterpret_cast<const DomainView<Anywhere>*>(this);
  }
};

template<typename Space>
class Domain {
 public:
  static std::shared_ptr<Domain> make(const io::LearningSet&);

  template<typename OtherSpace> friend class Domain;

  template<typename OtherSpace, typename = std::enable_if_t<!std::is_same_v<OtherSpace, Space>>>
  std::shared_ptr<Domain<OtherSpace>> clone_to() const {
    return std::make_shared<Domain<OtherSpace>>(
      typename Domain<OtherSpace>::Privacy(),
      _categories_count,
      _criteria_count,
      _learning_alternatives_count,
      _learning_alternatives.template clone_to<OtherSpace>(),
      _learning_assignments.template clone_to<OtherSpace>());
  }

 public:
  DomainView<Space> get_view() const;

  // Constructor has to be public for std::make_shared, but we want to make it unaccessible to external code,
  // so we make that constructor require a private structure.
 private:
  struct Privacy {};

 public:
  Domain(Privacy, uint, uint, uint, Array2D<Space, float>&&, Array1D<Space, uint>&&);

 private:
  uint _categories_count;
  uint _criteria_count;
  uint _learning_alternatives_count;
  Array2D<Space, float> _learning_alternatives;
  Array1D<Space, uint> _learning_assignments;
};

/*
The variables of the problem: the models being trained
*/
template<typename Space>
struct ModelsView {
  DomainView<Space> domain;

  const uint models_count;

  const ArrayView1D<Space, uint> initialization_iteration_indexes;
  // Index: index of model, from `0` to `models_count - 1`

  const ArrayView2D<Space, float> weights;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Compared to their description in the thesis, weights are denormalized:
  // - their sum is not constrained to be 1
  // - we don't store the threshold; we assume it's always 1
  // - this approach corresponds to dividing the weights and threshold as defined in the thesis by the threshold
  // - it simplifies the implementation because it removes the sum constraint and the threshold variables

  const ArrayView3D<Space, float> profiles;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of category below profile, from `0` to `domain.categories_count - 2`
  // Third index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance

  template<typename S = Space, typename = std::enable_if_t<!std::is_same_v<S, Anywhere>>>
  operator const ModelsView<Anywhere>&() const {
    return *reinterpret_cast<const ModelsView<Anywhere>*>(this);
  }
};


template<typename Space>
class Models {
 public:
  static std::shared_ptr<Models> make(std::shared_ptr<Domain<Space>>, const std::vector<io::Model>&);
  static std::shared_ptr<Models> make(std::shared_ptr<Domain<Space>>, uint models_count);

  io::Model unmake_one(uint model_index) const;
  std::vector<io::Model> unmake() const;

  template<typename OtherSpace> friend class Models;

  template<typename OtherSpace, typename = std::enable_if_t<!std::is_same_v<OtherSpace, Space>>>
  std::shared_ptr<Models<OtherSpace>> clone_to(std::shared_ptr<Domain<OtherSpace>> other_domain) const {
    return std::make_shared<Models<OtherSpace>>(
      typename Models<OtherSpace>::Privacy(),
      other_domain,
      _models_count,
      _initialization_iteration_indexes.template clone_to<OtherSpace>(),
      _weights.template clone_to<OtherSpace>(),
      _profiles.template clone_to<OtherSpace>());
  }

 public:
  std::shared_ptr<Domain<Space>> get_domain() { return _domain; }
  ModelsView<Space> get_view() const;  // @todo Remove const

 private:
  struct Privacy {};

 public:
  Models(
    Privacy,
    std::shared_ptr<Domain<Space>>,
    uint,
    Array1D<Space, uint>&&,
    Array2D<Space, float>&&,
    Array3D<Space, float>&&);

 private:
  std::shared_ptr<Domain<Space>> _domain;
  uint _models_count;
  Array1D<Space, uint> _initialization_iteration_indexes;
  Array2D<Space, float> _weights;
  Array3D<Space, float> _profiles;
};

}  // namespace ppl

#endif  // PROBLEM_HPP_
#line 1 "observe.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OBSERVE_HPP_
#define OBSERVE_HPP_

// Commented during copy-paste from PPL: #include "problem.hpp"


namespace ppl {

class LearningObserver {
 public:
  virtual ~LearningObserver() {}

  virtual void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) = 0;
};

}  // namespace ppl

#endif  // OBSERVE_HPP_
#line 1 "initialize-profiles.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef INITIALIZE_PROFILES_HPP_
#define INITIALIZE_PROFILES_HPP_

#include <memory>
#include <vector>

// Commented during copy-paste from PPL: #include "problem.hpp"


namespace ppl {

class ProfilesInitializationStrategy {
 public:
  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    std::shared_ptr<Models<Host>> models,
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end) = 0;
};

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_HPP_
#line 1 "randomness.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef RANDOMNESS_HPP_
#define RANDOMNESS_HPP_


#include <omp.h>

#include <map>
#include <optional>
#include <random>
#include <vector>
#include <cassert>


/*
A source of randomness.
*/
class Random {
 public:
  explicit Random(int seed) : _gen(omp_get_max_threads()) {
    #pragma omp parallel
    {
      urbg().seed(seed * (omp_get_thread_num() + 1));
    }
  }

  // Non-copyable
  Random(const Random&) = delete;
  Random& operator=(const Random&) = delete;
  // Could be made movable if needed
  Random(Random&&) = delete;
  Random& operator=(Random&&) = delete;

  float uniform_float(const float min, const float max) const {
    float v = max;

    do {
      v = std::uniform_real_distribution<float>(min, max)(urbg());
    } while (v == max);

    return v;
  }

  uint uniform_int(const uint min, const uint max) const {
    return std::uniform_int_distribution<unsigned int>(min, max - 1)(urbg());
  }

  std::mt19937& urbg() const {
    const unsigned thread_index = omp_get_thread_num();
    assert(thread_index < _gen.size());
    return _gen[thread_index];
  }

 private:
  mutable std::vector<std::mt19937> _gen;
};

/*
Pick random values from a finite set with given probabilities
(a discrete distribution with arbitrary values).
*/
template<typename T>
class ProbabilityWeightedGenerator {
  ProbabilityWeightedGenerator(const std::vector<T>& values, const std::vector<double>& probabilities) :
    _values(values),
    _distribution(probabilities.begin(), probabilities.end())
  {}

 public:
  static ProbabilityWeightedGenerator make(std::map<T, double> value_probabilities) {
    std::vector<T> values;
    values.reserve(value_probabilities.size());
    std::vector<double> probabilities;
    probabilities.reserve(value_probabilities.size());
    for (auto value_probability : value_probabilities) {
      values.push_back(value_probability.first);
      probabilities.push_back(value_probability.second);
    }
    return ProbabilityWeightedGenerator(values, probabilities);
  }

  std::map<T, double> get_value_probabilities() {
    std::map<T, double> value_probabilities;
    auto probabilities = _distribution.probabilities();
    const uint size = _values.size();
    assert(probabilities.size() == size);
    for (uint i = 0; i != size; ++i) {
      value_probabilities[_values[i]] = probabilities[i];
    }
    return value_probabilities;
  }

  template<typename Generator>
  T operator()(Generator& gen) const {  // NOLINT(runtime/references)
    const uint index = _distribution(gen);
    assert(index < _values.size());
    return _values[index];
  }

 private:
  std::vector<T> _values;
  mutable std::discrete_distribution<uint> _distribution;
};


#endif  // RANDOMNESS_HPP_
#line 1 "improve-profiles.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_HPP_
#define IMPROVE_PROFILES_HPP_

#include <memory>

// Commented during copy-paste from PPL: #include "problem.hpp"


namespace ppl {

class ProfilesImprovementStrategy {
 public:
  virtual ~ProfilesImprovementStrategy() {}

  // @todo Accept a plain pointer instead of a shared_ptr
  // We don't do memory management here and should not force it on the caller
  // Make sure to apply the same change to all other places appropriate
  virtual void improve_profiles(std::shared_ptr<Models<Host>>) = 0;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HPP_
#line 1 "improve-profiles/accuracy-heuristic-gpu.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_ACCURACY_HEURISTIC_GPU_HPP_
#define IMPROVE_PROFILES_ACCURACY_HEURISTIC_GPU_HPP_

#include <memory>

// Commented during copy-paste from PPL: #include <chrones.hpp>

// Commented during copy-paste from PPL: #include "../improve-profiles.hpp"
// Commented during copy-paste from PPL: #include "../randomness.hpp"


namespace ppl {

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ImproveProfilesWithAccuracyHeuristicOnGpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicOnGpu(
      const Random& random,
      std::shared_ptr<Models<Device>> device_models) :
    _random(random),
    _device_models(device_models) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  const Random& _random;
  std::shared_ptr<Models<Device>> _device_models;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_ACCURACY_HEURISTIC_GPU_HPP_
#line 1 "optimize-weights.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_HPP_
#define OPTIMIZE_WEIGHTS_HPP_

#include <memory>

// Commented during copy-paste from PPL: #include "problem.hpp"


namespace ppl {

class WeightsOptimizationStrategy {
 public:
  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights(std::shared_ptr<Models<Host>>) = 0;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_HPP_
#line 1 "terminate.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_HPP_
#define TERMINATE_HPP_

// Commented during copy-paste from PPL: #include "uint.hpp"


namespace ppl {

class TerminationStrategy {
 public:
  virtual ~TerminationStrategy() {}

  virtual bool terminate(uint iteration_index, uint best_accuracy) = 0;
};

}  // namespace ppl

#endif  // TERMINATE_HPP_
#line 1 "learning.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef LEARNING_HPP_
#define LEARNING_HPP_

#include <memory>
#include <vector>

// Commented during copy-paste from PPL: #include "improve-profiles.hpp"
// Commented during copy-paste from PPL: #include "initialize-profiles.hpp"
// Commented during copy-paste from PPL: #include "observe.hpp"
// Commented during copy-paste from PPL: #include "optimize-weights.hpp"
// Commented during copy-paste from PPL: #include "terminate.hpp"


namespace ppl {

struct LearningResult {
  LearningResult(io::Model model, uint accuracy) : best_model(model), best_model_accuracy(accuracy) {}

  io::Model best_model;
  uint best_model_accuracy;
};

// @todo Find a good default value. How?
const uint default_models_count = 9;

LearningResult perform_learning(
  std::shared_ptr<Models<Host>> host_models,
  std::vector<std::shared_ptr<LearningObserver>> observers,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
);

}  // namespace ppl

#endif  // LEARNING_HPP_
