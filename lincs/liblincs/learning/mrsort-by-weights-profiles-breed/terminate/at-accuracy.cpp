// Copyright 2023-2024 Vincent Jacques

#include "at-accuracy.hpp"


namespace lincs {

bool TerminateAtAccuracy::terminate() {
  return learning_data.get_best_accuracy() >= target_accuracy;
}

}  // namespace lincs
