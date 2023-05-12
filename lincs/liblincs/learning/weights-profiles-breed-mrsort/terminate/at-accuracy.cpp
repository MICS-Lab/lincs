// Copyright 2023 Vincent Jacques

#include "at-accuracy.hpp"


namespace lincs {

bool TerminateAtAccuracy::terminate(unsigned /*iteration_index*/, unsigned best_accuracy) {
  return best_accuracy >= _target_accuracy;
}

}  // namespace lincs
