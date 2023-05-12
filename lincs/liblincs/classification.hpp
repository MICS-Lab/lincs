#ifndef LINCS_CLASSIFICATION_HPP
#define LINCS_CLASSIFICATION_HPP

#include "io.hpp"


namespace lincs {

struct ClassificationResult {
  unsigned unchanged;
  unsigned changed;
};

ClassificationResult classify_alternatives(const Domain&, const Model&, Alternatives*);

}  // namespace lincs

#endif  // LINCS_CLASSIFICATION_HPP
