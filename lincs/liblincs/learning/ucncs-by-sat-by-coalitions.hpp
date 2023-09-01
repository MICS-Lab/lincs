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
    coalitions_count(1 << criteria_count),
    alternatives_count(learning_set.alternatives.size()),
    unique_values(),
    better(),
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
  // @todo(Performance, later) Dematerialize 'all_coalitions':
  // use a more abstract class that can be used in place of the current std::vector<boost::dynamic_bitset<>>
  // Same in "max-SAT by coalitions"
  typedef boost::dynamic_bitset<> Coalition;
  std::vector<Coalition> all_coalitions;
  const unsigned alternatives_count;
  std::vector<std::vector<float>> unique_values;
  // better[criterion_index][boundary_index][value_index]: value is better than profile on criterion
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> better;
  // sufficient[coalition.to_ulong()]: coalition is sufficient
  std::vector<typename SatProblem::variable_type> sufficient;
  SatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
