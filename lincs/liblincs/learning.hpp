// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING_HPP
#define LINCS__LEARNING_HPP

#include "learning/exception.hpp"
#include "learning/mrsort-by-weights-profiles-breed.hpp"
#include "learning/mrsort-by-weights-profiles-breed/breed/reinitialize-least-accurate.hpp"
#include "learning/mrsort-by-weights-profiles-breed/improve-profiles/accuracy-heuristic-on-cpu.hpp"
#include "learning/mrsort-by-weights-profiles-breed/improve-profiles/accuracy-heuristic-on-gpu.hpp"
#include "learning/mrsort-by-weights-profiles-breed/initialize-profiles/probabilistic-maximal-discrimination-power-per-criterion.hpp"
#include "learning/mrsort-by-weights-profiles-breed/optimize-weights/linear-program.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/after-iterations.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/after-iterations-without-progress.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/after-seconds.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/after-seconds-without-progress.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/at-accuracy.hpp"
#include "learning/mrsort-by-weights-profiles-breed/terminate/composite.hpp"
#include "learning/ucncs-by-max-sat-by-coalitions.hpp"
#include "learning/ucncs-by-max-sat-by-separation.hpp"
#include "learning/ucncs-by-sat-by-coalitions.hpp"
#include "learning/ucncs-by-sat-by-separation.hpp"
#include "linear-programming/alglib.hpp"
#include "linear-programming/glop.hpp"
#include "sat/eval-max-sat.hpp"
#include "sat/minisat.hpp"

namespace lincs {
  typedef MaxSatCoalitionsUcncsLearning<EvalmaxsatMaxSatProblem> LearnUcncsByMaxSatByCoalitionsUsingEvalmaxsat;
  typedef MaxSatSeparationUcncsLearning<EvalmaxsatMaxSatProblem> LearnUcncsByMaxSatBySeparationUsingEvalmaxsat;
  typedef OptimizeWeightsUsingLinearProgram<AlglibLinearProgram> OptimizeWeightsUsingAlglib;
  typedef OptimizeWeightsUsingLinearProgram<GlopLinearProgram> OptimizeWeightsUsingGlop;
  typedef SatCoalitionsUcncsLearning<MinisatSatProblem> LearnUcncsBySatByCoalitionsUsingMinisat;
  typedef SatSeparationUcncsLearning<MinisatSatProblem> LearnUcncsBySatBySeparationUsingMinisat;
}  // namespace lincs

#endif  // LINCS__LEARNING_HPP
