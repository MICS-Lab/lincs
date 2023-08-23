// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
#define LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP

#include "../io.hpp"


namespace lincs {

template<typename SatProblem>
class SatCoalitionsUcncsLearning {
 public:
  SatCoalitionsUcncsLearning(const Problem& problem_, const Alternatives& learning_set_) :
    problem(problem_),
    learning_set(learning_set_),
    criteria_count(problem.criteria.size()),
    categories_count(problem.categories.size()),
    boundaries_count(categories_count - 1),
    subsets_count(1 << criteria_count),
    alternatives_count(learning_set.alternatives.size()),
    unique_values(),
    above(),
    sufficient(),
    sat()
  {}

  // Not copyable
  SatCoalitionsUcncsLearning(const SatCoalitionsUcncsLearning&) = delete;
  SatCoalitionsUcncsLearning& operator=(const SatCoalitionsUcncsLearning&) = delete;
  // Could be made movable if needed
  SatCoalitionsUcncsLearning(SatCoalitionsUcncsLearning&&) = delete;
  SatCoalitionsUcncsLearning& operator=(SatCoalitionsUcncsLearning&&) = delete;

 public:
  Model perform();

 private:
  void sort_values();
  void create_variables();
  void add_structural_constraints();
  void add_learning_set_constraints();
  Model decode(const std::vector<bool>& solution);

 private:
  const Problem& problem;
  const Alternatives& learning_set;
  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned boundaries_count;
  const unsigned subsets_count;
  const unsigned alternatives_count;
  std::vector<std::vector<float>> unique_values;
  // above[criterion_index][boundary_index][value_index]: value is above profile on criterion
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> above;
  // A subset of criteria (i.e. a coalition) is represented as an unsigned int where
  // bit i is set if and only if criteria i is in the subset
  // sufficient[subset]: subset is a sufficient coalition
  std::vector<typename SatProblem::variable_type> sufficient;
  SatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
