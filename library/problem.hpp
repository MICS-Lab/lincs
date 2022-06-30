// Copyright 2021-2022 Vincent Jacques

#ifndef PROBLEM_HPP_
#define PROBLEM_HPP_

#include <memory>
#include <vector>

#include "cuda-utils.hpp"
#include "io.hpp"
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

  ArrayView2D<Anywhere, const float> learning_alternatives;
  // First index: index of criterion, from `0` to `criteria_count - 1`
  // Second index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Values are pre-normalized on each criterion so that the possible values are from `0.0` to `1.0`.
  // @todo Can we relax this assumption?
  //  - going from `-infinity` to `+infinity` might be possible
  //  - or we can extract the smallest and greatest value of each criterion on all the alternatives
  //  - to handle criterion where a lower value is better, we'd need to store an additional boolean indicator

  ArrayView1D<Anywhere, const uint> learning_assignments;
  // Index: index of alternative, from `0` to `learning_alternatives_count - 1`
  // Possible values: from `0` to `categories_count - 1`
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
  DomainView get_view() const;

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
struct ModelsView {
  DomainView domain;

  const uint models_count;

  const ArrayView1D<Anywhere, uint> initialization_iteration_indexes;
  // Index: index of model, from `0` to `models_count - 1`

  const ArrayView2D<Anywhere, float> weights;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Compared to their description in the thesis, weights are denormalized:
  // - their sum is not constrained to be 1
  // - we don't store the threshold; we assume it's always 1
  // - this approach corresponds to dividing the weights and threshold as defined in the thesis by the threshold
  // - it simplifies the implementation because it removes the sum constraint and the threshold variables

  const ArrayView3D<Anywhere, float> profiles;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of category below profile, from `0` to `domain.categories_count - 2`
  // Third index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
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
  ModelsView get_view() const;  // @todo Remove const

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

  friend void replicate_models(const Models<Host>&, Models<Device>*);
  friend void replicate_profiles(const Models<Device>&, Models<Host>*);

 private:
  std::shared_ptr<Domain<Space>> _domain;
  uint _models_count;
  Array1D<Space, uint> _initialization_iteration_indexes;
  Array2D<Space, float> _weights;
  Array3D<Space, float> _profiles;
};

// Utility function to replicate weights (computed on the host) and
// profiles (re-initialized on the host) onto the device
void replicate_models(const Models<Host>&, Models<Device>*);

// Utility function to replicate profiles (computed on the device) onto the host
void replicate_profiles(const Models<Device>&, Models<Host>*);

struct CandidatesView {
  DomainView domain;

  ArrayView1D<Anywhere, const uint> candidates_counts;
  // Index: index of criterion, from `0` to `domain.criteria_count - 1`

  ArrayView2D<Anywhere, const float> candidates;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of candidate, from `0` to `candidates_counts[crit_index] - 1`
};

template<typename Space>
class Candidates {
 public:
  static std::shared_ptr<Candidates> make(std::shared_ptr<Domain<Space>>);

  // Non-copyable
  Candidates(const Candidates&) = delete;
  Candidates& operator=(const Candidates&) = delete;

  // Non-movable
  Candidates(Candidates&&) = delete;
  Candidates& operator=(Candidates&&) = delete;

  template<typename OtherSpace> friend class Candidates;

  template<typename OtherSpace, typename = std::enable_if_t<!std::is_same_v<OtherSpace, Space>>>
  std::shared_ptr<Candidates<OtherSpace>> clone_to(std::shared_ptr<Domain<OtherSpace>> other_domain) const {
    return std::make_shared<Candidates<OtherSpace>>(
      typename Candidates<OtherSpace>::Privacy(),
      other_domain,
      _candidates_counts.template clone_to<OtherSpace>(),
      _max_candidates_count,
      _candidates.template clone_to<OtherSpace>());
  }

 public:
  CandidatesView get_view() const;

 private:
  struct Privacy {};

 public:
  Candidates(Privacy, std::shared_ptr<Domain<Space>>, Array1D<Space, uint>&&, uint, Array2D<Space, float>&&);

 private:
  std::shared_ptr<Domain<Space>> _domain;
  Array1D<Space, uint> _candidates_counts;
  const uint _max_candidates_count;
  Array2D<Space, float> _candidates;
};

}  // namespace ppl

#endif  // PROBLEM_HPP_
