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
  static Domain make(const std::vector<LearningAlternative>& learning_alternatives);

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

#endif  // IMPROVE_PROFILES_HPP_
