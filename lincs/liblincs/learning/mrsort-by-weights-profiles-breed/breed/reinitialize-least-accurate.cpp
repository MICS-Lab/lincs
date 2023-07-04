// Copyright 2023 Vincent Jacques

#include "reinitialize-least-accurate.hpp"


namespace lincs {

void ReinitializeLeastAccurate::breed() {
  profiles_initialization_strategy.initialize_profiles(0, count);
}

}  // namespace lincs
