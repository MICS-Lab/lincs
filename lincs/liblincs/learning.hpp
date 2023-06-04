// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING_HPP
#define LINCS__LEARNING_HPP

#include "learning/weights-profiles-breed-mrsort.hpp"
#include "learning/weights-profiles-breed-mrsort/improve-profiles/accuracy-heuristic-on-cpu.hpp"
#include "learning/weights-profiles-breed-mrsort/improve-profiles/accuracy-heuristic-on-gpu.hpp"
#include "learning/weights-profiles-breed-mrsort/initialize-profiles/probabilistic-maximal-discrimination-power-per-criterion.hpp"
#include "learning/weights-profiles-breed-mrsort/optimize-weights/linear-program.hpp"
#include "learning/weights-profiles-breed-mrsort/terminate/at-accuracy.hpp"
#include "linear-programming/alglib.hpp"
#include "linear-programming/glop.hpp"

namespace lincs {
  typedef OptimizeWeightsUsingLinearProgram<AlglibLinearProgram> OptimizeWeightsUsingAlglib;
  typedef OptimizeWeightsUsingLinearProgram<GlopLinearProgram> OptimizeWeightsUsingGlop;
}  // namespace lincs

#endif  // LINCS__LEARNING_HPP
