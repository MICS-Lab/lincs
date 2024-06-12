// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
#define LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP

#include "../io.hpp"
#include "pre-processing.hpp"


namespace lincs {

template<typename SatProblem>
class SatCoalitionsUcncsLearning {
 public:
  template<class... U>
  SatCoalitionsUcncsLearning(const Problem& problem, const Alternatives& learning_set_, U&&... u) :
    #ifndef NDEBUG
    input_problem(problem),
    input_learning_set(learning_set_),
    #endif
    learning_set(problem, learning_set_),
    coalitions_count(1 << learning_set.criteria_count),
    accepted(),
    sufficient(),
    sat(std::forward<U>(u)...)
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
  #ifndef NDEBUG
  const Problem& input_problem;
  const Alternatives& input_learning_set;
  #endif
  PreProcessedLearningSet learning_set;
  const unsigned coalitions_count;
  // @todo(Performance, later) Dematerialize 'all_coalitions':
  // use a more abstract class that can be used in place of the current std::vector<boost::dynamic_bitset<>>
  // Same in "max-SAT by coalitions"
  typedef boost::dynamic_bitset<> Coalition;
  std::vector<Coalition> all_coalitions;
  // accepted[criterion_index][boundary_index][value_rank]: value is accepted by boundary on criterion (above profile for monotonous criteria, inside interval for single-peaked criteria)
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> accepted;
  // sufficient[coalition.to_ulong()]: coalition is sufficient
  std::vector<typename SatProblem::variable_type> sufficient;
  SatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_SAT_BY_COALITIONS_HPP
