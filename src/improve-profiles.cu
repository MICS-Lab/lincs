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
