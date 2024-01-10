// Copyright 2023-2024 Vincent Jacques

#include "minisat.hpp"

#include "../chrones.hpp"


namespace lincs {

std::optional<std::vector<bool>> MinisatSatProblem::solve() {
  CHRONE();

  Minisat::vec<Minisat::Lit> dummy;
  const auto ret = solver.solveLimited(dummy);

  if (ret == Minisat::l_True) {
    std::vector<bool> solution(solver.nVars() + 1);
    for (int i = 0; i < solver.nVars(); ++i) {
      solution[i + 1] = solver.model[i] == Minisat::l_True;
    }

    return solution;
  } else {
    return std::nullopt;
  }
}

}  // namespace lincs
