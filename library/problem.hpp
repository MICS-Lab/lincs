// Copyright 2021-2022 Vincent Jacques

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include <vector>

#include "cuda-utils.hpp"
#include "io.hpp"
#include "matrix-view.hpp"
#include "uint.hpp"


namespace ppl {

/*
The constants of the problem, i.e. the sizes of the domain, and the learning set.

@todo Split Domain and DomainView class into Domain proper (sizes, labels, etc.) and LearningSet (classified alternatives)
*/
struct DomainView {
  const uint categories_count;
  const uint criteria_count;
  const uint learning_alternatives_count;

  MatrixView2D<const float> learning_alternatives;
  // First index: index of criterion, from `0` to `criteria_count - 1`
  // Second index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Values are pre-normalized on each criterion so that the possible values are from `0.0` to `1.0`.
  // @todo Can we relax this assumption?
  //  - going from `-infinity` to `+infinity` might be possible
  //  - or we can extract the smallest and greatest value of each criterion on all the alternatives
  //  - to handle criterion where a lower value is better, we'd need to store an aditional boolean indicator

  MatrixView1D<const uint> learning_assignments;
  // Index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // Possible values: from `0` to `categories_count - 1`
};

template<typename Space>
class Domain {
 public:
  static Domain make(const io::LearningSet&);
  ~Domain();

  // Non-copyable
  Domain(const Domain&) = delete;
  Domain& operator=(const Domain&) = delete;

  // Movable (not yet implemented)
  Domain(Domain&&);
  Domain& operator=(Domain&&);

  template<typename OtherSpace> friend class Domain;

  template<typename OtherSpace, typename = std::enable_if_t<!std::is_same_v<OtherSpace, Space>>>
  Domain<OtherSpace> clone_to() const {
    return Domain<OtherSpace>(
      categories_count,
      criteria_count,
      learning_alternatives_count,
      FromTo<Space, OtherSpace>::clone(criteria_count * learning_alternatives_count, learning_alternatives),
      FromTo<Space, OtherSpace>::clone(learning_alternatives_count, learning_assignments));
  }

 public:
  DomainView get_view() const;

 private:
  Domain(uint, uint, uint, float*, uint*);

 private:
  uint categories_count;
  uint criteria_count;
  uint learning_alternatives_count;
  float* learning_alternatives;
  uint* learning_assignments;
};

/*
The variables of the problem: the models being trained
*/
struct ModelsView {
  DomainView domain;

  const uint models_count;

  const MatrixView1D<uint> initialization_iteration_indexes;
  // Index: index of model, from `0` to `models_count - 1`

  const MatrixView2D<float> weights;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Compared to their description in the thesis, weights are denormalized:
  // - their sum is not constrained to be 1
  // - we don't store the threshold; we assume it's always 1
  // - this approach corresponds to dividing the weights and threshold as defined in the thesis by the threshold
  // - it simplifies the implementation because it removes the sum constraint and the threshold variables

  const MatrixView3D<float> profiles;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of category below profile, from `0` to `domain.categories_count - 2`
  // Third index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
};

template<typename Space>
class Models {
 public:
  static Models make(
    const Domain<Space>& domain /* Stored by reference. Don't let the Domain be destructed before the Models. */,
    const std::vector<io::Model>& models);
  static Models make(
    const Domain<Space>& domain /* Stored by reference. Don't let the Domain be destructed before the Models. */,
    uint models_count);
  ~Models();

  io::Model unmake_one(uint model_index) const;
  std::vector<io::Model> unmake() const;

  // Non-copyable
  Models(const Models&) = delete;
  Models& operator=(const Models&) = delete;

  // Movable (not yet implemented)
  Models(Models&&);
  Models& operator=(Models&&);

  template<typename OtherSpace> friend class Models;

  template<typename OtherSpace, typename = std::enable_if_t<!std::is_same_v<OtherSpace, Space>>>
  Models<OtherSpace> clone_to(const Domain<OtherSpace>& domain) const {
    DomainView domain_view = domain.get_view();
    return Models<OtherSpace>(
      domain,
      models_count,
      FromTo<Space, OtherSpace>::clone(models_count, initialization_iteration_indexes),
      FromTo<Space, OtherSpace>::clone(domain_view.criteria_count * models_count, weights),
      FromTo<Space, OtherSpace>::clone(
        domain_view.criteria_count * (domain_view.categories_count - 1) * models_count,
        profiles));
  }

 public:
  ModelsView get_view() const;  // @todo Remove const

 private:
  Models(const Domain<Space>&, uint, uint*, float*, float*);

  friend void replicate_models(const Models<Host>&, Models<Device>*);
  friend void replicate_profiles(const Models<Device>&, Models<Host>*);

 private:
  const Domain<Space>& domain;
  uint models_count;
  uint* initialization_iteration_indexes;
  float* weights;
  float* profiles;
};

// Utility function to replicate weights (computed on the host) and
// profiles (re-initialized on the host) onto the device
void replicate_models(const Models<Host>&, Models<Device>*);

// Utility function to replicate profiles (computed on the device) onto the host
void replicate_profiles(const Models<Device>&, Models<Host>*);

}  // namespace ppl

#endif  // PROBLEM_HPP_
