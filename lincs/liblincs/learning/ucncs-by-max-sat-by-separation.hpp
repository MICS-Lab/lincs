// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
#define LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP

#include "../io.hpp"
#include "pre-processing.hpp"


namespace lincs {

template<typename MaxSatProblem>
class MaxSatSeparationUcncsLearning {
 public:
  template<class... U>
  MaxSatSeparationUcncsLearning(const Problem& problem, const Alternatives& learning_set_, U&&... u) :
    #ifndef NDEBUG
    input_problem(problem),
    input_learning_set(learning_set_),
    #endif
    learning_set(problem, learning_set_),
    subgoal_weight(1),
    goal_weight(learning_set.boundaries_count * learning_set.alternatives_count),
    better_alternative_indexes(),
    worse_alternative_indexes(),
    accepted(),
    separates(),
    sat(std::forward<U>(u)...)
  {}

 public:
  Model perform();

 private:
  void sort_values();
  void partition_alternatives();
  void create_variables();
  void add_structural_constraints();
  void add_learning_set_constraints();
  Model decode(const std::vector<bool>& solution);

 private:
  #ifndef NDEBUG
  const Problem& input_problem;
  const Alternatives& input_learning_set;
  #endif
  PreprocessedLearningSet learning_set;
  const typename MaxSatProblem::weight_type subgoal_weight;
  const typename MaxSatProblem::weight_type goal_weight;
  // See more comments in 'ucncs-by-sat-by-coalitions.hpp' and 'ucncs-by-sat-by-separation.hpp'
  std::vector<std::vector<unsigned>> better_alternative_indexes;
  std::vector<std::vector<unsigned>> worse_alternative_indexes;
  std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>> accepted;
  std::vector<std::vector<std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>>>> separates;
  std::vector<typename MaxSatProblem::variable_type> correct;
  std::vector<std::vector<typename MaxSatProblem::variable_type>> proper;
  MaxSatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
