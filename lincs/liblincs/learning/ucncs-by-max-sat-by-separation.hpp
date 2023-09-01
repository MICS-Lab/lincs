// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP
#define LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_SEPARATION_HPP

#include "../io.hpp"


namespace lincs {

template<typename MaxSatProblem>
class MaxSatSeparationUcncsLearning {
 public:
  MaxSatSeparationUcncsLearning(const Problem& problem_, const Alternatives& learning_set_) :
    problem(problem_),
    learning_set(learning_set_),
    criteria_count(problem.criteria.size()),
    categories_count(problem.categories.size()),
    boundaries_count(categories_count - 1),
    alternatives_count(learning_set.alternatives.size()),
    subgoal_weight(1),
    goal_weight(boundaries_count * alternatives_count),
    unique_values(),
    better_alternative_indexes(),
    worse_alternative_indexes(),
    better(),
    separates(),
    sat()
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
  const Problem& problem;
  const Alternatives& learning_set;
  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned boundaries_count;
  const unsigned alternatives_count;
  const typename MaxSatProblem::weight_type subgoal_weight;
  const typename MaxSatProblem::weight_type goal_weight;
  std::vector<std::vector<float>> unique_values;
  // Alternatives better than category k
  std::vector<std::vector<unsigned>> better_alternative_indexes;
  // Alternatives in category k or worse
  std::vector<std::vector<unsigned>> worse_alternative_indexes;
  // better[criterion_index][boundary_index][value_index]: value is better profile on criterion
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
