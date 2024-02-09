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
    learning_set(problem, learning_set_),
    subgoal_weight(1),
    goal_weight(learning_set.boundaries_count * learning_set.alternatives_count),
    better_alternative_indexes(),
    worse_alternative_indexes(),
    better(),
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
  PreProcessedLearningSet learning_set;
  const typename MaxSatProblem::weight_type subgoal_weight;
  const typename MaxSatProblem::weight_type goal_weight;
  // Alternatives better than category k
  std::vector<std::vector<unsigned>> better_alternative_indexes;
  // Alternatives in category k or worse
  std::vector<std::vector<unsigned>> worse_alternative_indexes;
  // better[criterion_index][boundary_index][value_rank]: value is better profile on criterion
  std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>> better;
  // separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index]:
  // criterion separates alternatives 'good' and 'bad' with regards to profiles 'a' and 'b'
  std::vector<std::vector<std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>>>> separates;
  // correct[alternative_index]: alternative is correctly classified
  std::vector<typename MaxSatProblem::variable_type> correct;
  // proper[alternative_index][boundary_index]: alternative is properly classified by the 2-categories model defined by the boundary
  std::vector<std::vector<typename MaxSatProblem::variable_type>> proper;
  MaxSatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
