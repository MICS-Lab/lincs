// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP
#define LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP

#include "../io.hpp"


namespace lincs {

template<typename MaxSatProblem>
class MaxSatCoalitionsUcncsLearning {
 public:
  MaxSatCoalitionsUcncsLearning(const Problem& problem_, const Alternatives& learning_set_) :
    problem(problem_),
    learning_set(learning_set_),
    criteria_count(problem.criteria.size()),
    categories_count(problem.categories.size()),
    boundaries_count(categories_count - 1),
    coalitions_count(1 << criteria_count),
    alternatives_count(learning_set.alternatives.size()),
    goal_weight(1),
    unique_values(),
    better(),
    sufficient(),
    sat()
  {}

  // Not copyable
  MaxSatCoalitionsUcncsLearning(const MaxSatCoalitionsUcncsLearning&) = delete;
  MaxSatCoalitionsUcncsLearning& operator=(const MaxSatCoalitionsUcncsLearning&) = delete;
  // Could be made movable if needed
  MaxSatCoalitionsUcncsLearning(MaxSatCoalitionsUcncsLearning&&) = delete;
  MaxSatCoalitionsUcncsLearning& operator=(MaxSatCoalitionsUcncsLearning&&) = delete;

 public:
  Model perform();

 private:
  void sort_values();
  void create_all_coalitions();
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
  const unsigned coalitions_count;
  typedef boost::dynamic_bitset<> Coalition;
  std::vector<Coalition> all_coalitions;
  const unsigned alternatives_count;
  const typename MaxSatProblem::weight_type goal_weight;
  std::vector<std::vector<float>> unique_values;
  // better[criterion_index][boundary_index][value_index]: value is better than profile on criterion
  std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>> better;
  // sufficient[coalition.to_ulong()]: coalition is sufficient
  std::vector<typename MaxSatProblem::variable_type> sufficient;
  // correct[alternative_index]: alternative is correctly classified
  std::vector<typename MaxSatProblem::variable_type> correct;
  MaxSatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP
