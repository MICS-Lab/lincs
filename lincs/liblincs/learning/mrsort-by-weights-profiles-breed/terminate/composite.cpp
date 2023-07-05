// Copyright 2023 Vincent Jacques

#include "composite.hpp"


namespace lincs {

bool TerminateWhenAny::terminate() {
  for (auto termination_strategy : termination_strategies) {
    if (termination_strategy->terminate()) {
      return true;
    }
  }

  return false;
}

}  // namespace lincs
