// Copyright 2023 Vincent Jacques

#ifndef LINCS__CLASSIFICATION_HPP
#define LINCS__CLASSIFICATION_HPP

#include "io.hpp"


namespace lincs {

bool better_or_equal(Criterion::PreferenceDirection preference_direction, float lhs, float rhs);

struct ClassificationResult {
  unsigned unchanged;
  unsigned changed;
};

ClassificationResult classify_alternatives(const Problem&, const Model&, Alternatives*);

}  // namespace lincs

#endif  // LINCS__CLASSIFICATION_HPP
