// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
#define LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP

#include "../io.hpp"


namespace lincs {

template<typename SatProblem>
class MaxSatSeparationUcncsLearning {
 public:
  MaxSatSeparationUcncsLearning(const Problem& problem_, const Alternatives& learning_set_) : problem(problem_), learning_set(learning_set_) {}

 public:
  Model perform();

 private:
  const Problem& problem;
  const Alternatives& learning_set;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
