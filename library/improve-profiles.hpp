// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_PROFILES_HPP_
#define IMPROVE_PROFILES_HPP_

/* Algorithm 4 from https://tel.archives-ouvertes.fr/tel-01370555/document
Constants:
  - description of criteria:
    - number of criteria
    - for each criterion:
      - (optional) name
      - (optional: can be rescaled to 0 and 1) best and worst values
  - description of categories:
    - number of categories
    - for each category:
      - (optional) name
  - description of learning set of alternatives
    - number of alternatives
    - for each alternative:
      - value of all criteria
      - classification by the decision maker
  - description of models under training (constant part):
    - number of models
    - for each model:
      - the weight of each criterion
      - the coalition threshold

Variables (input and output, modified by this algorithm by side-effect):
  - description of models under training (variable part):
    - for each model:
      - value of all criteria for each profile
  - classification of each alternatives from the learning set by each model models
*/

#include <vector>

#include "io.hpp"
#include "cuda-utils.hpp"
#include "matrix-view.hpp"


namespace ppl::improve_profiles {

struct DomainView {
  const int categories_count;
  const int criteria_count;
  const int learning_alternatives_count;

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

  MatrixView1D<const int> learning_assignments;
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
  Domain(int, int, int, float*, int*);

 private:
  const int categories_count;
  const int criteria_count;
  const int learning_alternatives_count;
  float* const learning_alternatives;
  int* const learning_assignments;
};

struct ModelsView {
  DomainView domain;

  const int models_count;

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
  ~Models();

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
      FromTo<Space, OtherSpace>::clone(domain_view.criteria_count * models_count, weights),
      FromTo<Space, OtherSpace>::clone(
        domain_view.criteria_count * (domain_view.categories_count - 1) * models_count,
        profiles));
  }

 public:
  ModelsView get_view() const;  // @todo Remove const

 private:
  Models(const Domain<Space>&, int, float*, float*);

 private:
  const Domain<Space>& domain;
  const int models_count;
  float* const weights;
  float* const profiles;
};

int get_assignment(const Models<Host>&, int model_index, int alternative_index);

// Accuracy is returned as an integer between `0` and `models.domain.alternatives_count`.
// (To get the accuracy described in the thesis, it should be devided by `models.domain.alternatives_count`)
unsigned int get_accuracy(const Models<Host>&, int model_index);
unsigned int get_accuracy(const Models<Device>&, int model_index);

struct Desirability {
  int v = 0;
  int w = 0;
  int q = 0;
  int r = 0;
  int t = 0;

  __host__ __device__ float value() const {
    if (v + w + t + q + r == 0) {
      // The move has no impact. @todo What should its desirability be?
      return 0;
    } else {
      return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
    }
  }
};

void improve_profiles(Models<Host>*);
void improve_profiles(Models<Device>*);

}  // namespace ppl::improve_profiles

#endif  // IMPROVE_PROFILES_HPP_
