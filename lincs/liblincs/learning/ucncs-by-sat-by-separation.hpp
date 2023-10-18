// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__UCNCS_BY_SAT_BY_SEPARATION_HPP
#define LINCS__LEARNING__UCNCS_BY_SAT_BY_SEPARATION_HPP

#include "../io.hpp"
#include "pre-processing.hpp"


namespace lincs {

template<typename SatProblem>
class SatSeparationUcncsLearning {
 public:
  SatSeparationUcncsLearning(const Problem& problem, const Alternatives& learning_set_) :
    learning_set(problem, learning_set_),
    better_alternative_indexes(),
    worse_alternative_indexes(),
    better(),
    separates(),
    sat()
  {}

 public:
  Model perform();

 private:
  void partition_alternatives();
  void create_variables();
  void add_structural_constraints();
  void add_learning_set_constraints();
  Model decode(const std::vector<bool>& solution);

 private:
  PreProcessedLearningSet learning_set;
  // Alternatives better than category k
  std::vector<std::vector<unsigned>> better_alternative_indexes;
  // Alternatives in category k or worse
  std::vector<std::vector<unsigned>> worse_alternative_indexes;
  // better[criterion_index][boundary_index][value_rank]: value is better than profile on criterion
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> better;
  // separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index]:
  // criterion separates alternatives 'good' and 'bad' with regards to profiles 'a' and 'b'
  std::vector<std::vector<std::vector<std::vector<std::vector<typename SatProblem::variable_type>>>>> separates;
  SatProblem sat;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__UCNCS_BY_SAT_BY_SEPARATION_HPP
