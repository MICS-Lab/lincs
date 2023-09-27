// Copyright 2023 Vincent Jacques

#define EVALMAXSAT_IMPLEMENT
#include "eval-max-sat.hpp"

#include "../chrones.hpp"


namespace lincs {

std::optional<std::vector<bool>> EvalmaxsatMaxSatProblem::solve() {
  CHRONE();

  if (solver.solve()) {
    std::vector<bool> solution(variables.back() + 1);
    for (const int v : variables) {
      solution[v] = solver.getValue(v);
    }
    return solution;
  } else {
    return std::nullopt;
  }
}

}  // namespace lincs
