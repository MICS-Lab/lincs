// Copyright 2023 Vincent Jacques

#include "at-accuracy.hpp"


namespace lincs {

bool TerminateAtAccuracy::terminate() {
  return models.get_best_accuracy() >= target_accuracy;
}

}  // namespace lincs
