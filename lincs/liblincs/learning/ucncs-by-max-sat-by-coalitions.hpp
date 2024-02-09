// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP
#define LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP

#include "../io.hpp"
#include "pre-processing.hpp"


namespace lincs {

template<typename MaxSatProblem>
class MaxSatCoalitionsUcncsLearning {
 public:
  template<class... U>
  MaxSatCoalitionsUcncsLearning(const Problem& problem, const Alternatives& learning_set_, U&&... u) :
    learning_set(problem, learning_set_),
    coalitions_count(1 << learning_set.criteria_count),
    goal_weight(1),
    better(),
    sufficient(),
    sat(std::forward<U>(u)...)
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
  void create_all_coalitions();
  void create_variables();
  void add_structural_constraints();
  void add_learning_set_constraints();
  Model decode(const std::vector<bool>& solution);

 private:
  PreProcessedLearningSet learning_set;
  const unsigned coalitions_count;
  typedef boost::dynamic_bitset<> Coalition;
  std::vector<Coalition> all_coalitions;
  const typename MaxSatProblem::weight_type goal_weight;
  // better[criterion_index][boundary_index][value_rank]: value is better than profile on criterion
  std::vector<std::vector<std::vector<typename MaxSatProblem::variable_type>>> better;
  // sufficient[coalition.to_ulong()]: coalition is sufficient
  std::vector<typename MaxSatProblem::variable_type> sufficient;
  // correct[alternative_index]: alternative is correctly classified
  std::vector<typename MaxSatProblem::variable_type> correct;
  MaxSatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_MAX_SAT_BY_COALITIONS_HPP
