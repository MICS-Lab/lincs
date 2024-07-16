// Copyright 2023-2024 Vincent Jacques

#include "at-accuracy.hpp"


namespace lincs {

bool TerminateAtAccuracy::terminate() {
  return models_being_learned.get_best_accuracy() >= target_accuracy;
}

}  // namespace lincs
