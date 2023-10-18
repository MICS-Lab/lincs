// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
#define LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP

#include "../io.hpp"
#include "pre-processing.hpp"


namespace lincs {

template<typename SatProblem>
class SatCoalitionsUcncsLearning {
 public:
  SatCoalitionsUcncsLearning(const Problem& problem, const Alternatives& learning_set_) :
    learning_set(problem, learning_set_),
    coalitions_count(1 << learning_set.criteria_count),
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
  void create_all_coalitions();
  void create_variables();
  void add_structural_constraints();
  void add_learning_set_constraints();
  Model decode(const std::vector<bool>& solution);

 private:
  PreProcessedLearningSet learning_set;
  const unsigned coalitions_count;
  // @todo(Performance, later) Dematerialize 'all_coalitions':
  // use a more abstract class that can be used in place of the current std::vector<boost::dynamic_bitset<>>
  // Same in "max-SAT by coalitions"
  typedef boost::dynamic_bitset<> Coalition;
  std::vector<Coalition> all_coalitions;
  // better[criterion_index][boundary_index][value_rank]: value is better than profile on criterion
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> better;
  // sufficient[coalition.to_ulong()]: coalition is sufficient
  std::vector<typename SatProblem::variable_type> sufficient;
  SatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
