// Copyright 2023-2024 Vincent Jacques

#include "reinitialize-least-accurate.hpp"

#include "../../../chrones.hpp"


namespace lincs {

void ReinitializeLeastAccurate::breed() {
  CHRONE();

  profiles_initialization_strategy.initialize_profiles(0, count);
}

}  // namespace lincs
