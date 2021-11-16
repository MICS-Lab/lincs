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

#include "matrix.hpp"


struct LearningAlternative {
  std::vector<float> criteria;
  int assignment;
};

template<typename Space>
class Domain {
 public:
  static Domain make(int categories_count, const std::vector<LearningAlternative>& learning_alternatives);

 public:
  const int categories_count;
  const int criteria_count;
  const int learning_alternatives_count;

  Matrix2D<Space, float> learning_alternatives;
  // First index: index of criterion, from `0` to `criteria_count - 1`
  // Second index: index of alternative, from `0` to `alternatives_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Values are pre-normalized on each criterion so that the possible values are from `0.0` to `1.0`.
  // @todo Can we relax this assumption?
  //  - going from `-infinity` to `+infinity` might be possible
  //  - or we can extract the smallest and greatest value of each criterion on all the alternatives
  //  - to handle criterion where a lower value is better, we'd need to store an aditional boolean indicator

  Matrix1D<Space, int> learning_assignments;
  // Index: index of alternative, from `0` to `alternatives_count - 1`
  // Possible values: from `0` to `categories_count - 1`

 private:
  Domain(int, int, int, Matrix2D<Space, float>&&, Matrix1D<Space, int>&&);
};

struct Model {
  std::vector<std::vector<float>> profiles;
  std::vector<float> weights;
};

template<typename Space>
class Models {
 public:
  static Models make(
    const Domain<Space>& domain /* Stored by reference. Don't let the Domain be destructed before the Models. */,
    const std::vector<Model>& models);

 public:
  const Domain<Space>& domain;

  const int models_count;

  const Matrix2D<Space, float> weights;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance
  // Compared to their description in the thesis, weights are denormalized:
  // - their sum is not constrained to be 1
  // - we don't store the threshold; we assume it's always 1
  // - this approach corresponds to dividing the weights and threshold as defined in the thesis by the threshold
  // - it simplifies the implementation because it removes the sum constraint and the threshold variables

  Matrix3D<Space, float> profiles;
  // First index: index of criterion, from `0` to `domain.criteria_count - 1`
  // Second index: index of category below profile, from `0` to `domain.categories_count - 2`
  // Third index: index of model, from `0` to `models_count - 1`
  // (Warning: this might seem reversed and counter-intuitive for some mindsets)
  // @todo Investigate if this weird index order is actually improving performance

 private:
  Models(const Domain<Space>&, int, Matrix2D<Space, float>&&, Matrix3D<Space, float>&&);
};

template<typename Space>
int get_assignment(const Models<Space>& models, int model_index, int alternative_index);

// Accuracy is returned as an integer between `0` and `models.domain.alternatives_count`.
// (To get the accuracy described in the thesis, it should be devided by `models.domain.alternatives_count`)
template<typename Space>
int get_accuracy(const Models<Space>& models, int model_index);

#endif  // IMPROVE_PROFILES_HPP_
