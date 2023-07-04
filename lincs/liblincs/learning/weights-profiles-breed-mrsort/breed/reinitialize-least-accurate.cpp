// Copyright 2023 Vincent Jacques

#include "reinitialize-least-accurate.hpp"


namespace lincs {

void ReinitializeLeastAccurate::breed() {
  profiles_initialization_strategy.initialize_profiles(models.model_indexes.begin(), models.model_indexes.begin() + count);
}

}  // namespace lincs
