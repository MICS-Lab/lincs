// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__CLASSIFICATION_HPP
#define LINCS__CLASSIFICATION_HPP

#include "io.hpp"


namespace lincs {

template<typename T>
bool better_or_equal(Criterion::PreferenceDirection preference_direction, const T lhs,const T rhs) {
  switch (preference_direction) {
    case Criterion::PreferenceDirection::increasing:
      return lhs >= rhs;
    case Criterion::PreferenceDirection::decreasing:
      return lhs <= rhs;
  }
  unreachable();
}

struct ClassificationResult {
  unsigned unchanged;
  unsigned changed;
};

ClassificationResult classify_alternatives(const Problem&, const Model&, Alternatives*);

}  // namespace lincs

#endif  // LINCS__CLASSIFICATION_HPP
