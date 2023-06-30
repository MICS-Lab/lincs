// Copyright 2023 Vincent Jacques

#ifndef LINCS__SAT__MINISAT_HPP
#define LINCS__SAT__MINISAT_HPP

#include <vector>

#include "../vendored/minisat/simp/SimpSolver.h"


namespace lincs {

class MinisatSatProblem {
 public:
  typedef int variable_type;
  variable_type create_variable() {
    return solver.newVar() + 1;
  }

  void mark_all_variables_created() {}

 private:
  auto make_literal(variable_type v) {
    if (v > 0) {
      return Minisat::mkLit(v - 1);
    } else {
      assert(v < 0);
      return ~Minisat::mkLit(-v - 1);
    }
  }

 public:
  void add_clause(std::vector<variable_type> clause_) {
    Minisat::vec<Minisat::Lit> clause;
    for (auto variable : clause_) {
      clause.push(make_literal(variable));
    }
    solver.addClause(clause);
  }

  typedef void weight_type;

  auto solve() {
    solver.eliminate(true);
    assert(solver.okay());
    Minisat::vec<Minisat::Lit> dummy;
    solver.solveLimited(dummy);

    std::vector<bool> solution(solver.nVars() + 1);
    for (int i = 0; i < solver.nVars(); ++i) {
      solution[i + 1] = solver.model[i] == Minisat::l_True;
    }

    return solution;
  }

 private:
  Minisat::SimpSolver solver;
};

}  // namespace lincs

#endif  // LINCS__SAT__MINISAT_HPP
