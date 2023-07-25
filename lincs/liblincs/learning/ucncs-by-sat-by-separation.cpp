// Copyright 2023 Vincent Jacques

#include "ucncs-by-sat-by-separation.hpp"

#include <algorithm>
#include <map>
#include <type_traits>

#include "exception.hpp"
// @todo Let these two be included in alphabetical order
// Currently, swapping them results in the following error:
// could not convert ‘Glucose::lbool(2)’ from ‘Glucose::lbool’ to ‘Minisat::lbool’
#include "../sat/minisat.hpp"
#include "../sat/eval-max-sat.hpp"


namespace lincs {

template<typename SatProblem>
Model SatSeparationUcncsLearning<SatProblem>::perform() {
  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.categories.size();

  // @todo Extend this implementation to handle more than two categories
  if (categories_count != 2) {
    throw LearningFailureException();
  }

  std::vector<std::vector<float>> unique_values(criteria_count);
  for (auto alternative : learning_set.alternatives) {
    for (unsigned i = 0; i != criteria_count; ++i) {
      unique_values[i].push_back(alternative.profile[i]);
    }
  }
  for (auto& v : unique_values) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }

  std::vector<unsigned> good_alternative_indexes;
  std::vector<unsigned> bad_alternative_indexes;
  good_alternative_indexes.reserve(learning_set.alternatives.size());
  bad_alternative_indexes.reserve(learning_set.alternatives.size());
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives.size(); ++alternative_index) {
    if (learning_set.alternatives[alternative_index].category_index == 1) {
      good_alternative_indexes.push_back(alternative_index);
    } else {
      bad_alternative_indexes.push_back(alternative_index);
    }
  }

  SatProblem sat;

  // a[i][x]: value x is above profile on criterion i
  std::vector<std::vector<typename SatProblem::variable_type>> a(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    a[criterion_index].resize(unique_values[criterion_index].size());
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
      a[criterion_index][value_index] = sat.create_variable();
    }
  }

  // s[i][g][b]: criterion i separates alternatives g and b
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> s(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    s[criterion_index].resize(learning_set.alternatives.size());
    for (unsigned good_alternative_index : good_alternative_indexes) {
      s[criterion_index][good_alternative_index].resize(learning_set.alternatives.size());
      for (unsigned bad_alternative_index : bad_alternative_indexes) {
        s[criterion_index][good_alternative_index][bad_alternative_index] = sat.create_variable();
      }
    }
  }

  sat.mark_all_variables_created();

  // P1
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size() - 1; ++value_index) {
      sat.add_clause({
        -a[criterion_index][value_index],
        a[criterion_index][value_index + 1],
      });
    }
  }

  // P2
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned bad_alternative_index : bad_alternative_indexes) {
      const unsigned bad_value_index = std::distance(
        unique_values[criterion_index].begin(),
        std::lower_bound(
          unique_values[criterion_index].begin(),
          unique_values[criterion_index].end(),
          learning_set.alternatives[bad_alternative_index].profile[criterion_index]
        )
      );
      for (unsigned good_alternative_index : good_alternative_indexes) {
        sat.add_clause({
          -s[criterion_index][good_alternative_index][bad_alternative_index],
          -a[criterion_index][bad_value_index],
        });
      }
    }
  }

  // P3
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned good_alternative_index : good_alternative_indexes) {
      const unsigned good_value_index = std::distance(
        unique_values[criterion_index].begin(),
        std::lower_bound(
          unique_values[criterion_index].begin(),
          unique_values[criterion_index].end(),
          learning_set.alternatives[good_alternative_index].profile[criterion_index]
        )
      );
      for (unsigned bad_alternative_index : bad_alternative_indexes) {
        sat.add_clause({
          -s[criterion_index][good_alternative_index][bad_alternative_index],
          a[criterion_index][good_value_index],
        });
      }
    }
  }

  // P4
  for (unsigned good_alternative_index : good_alternative_indexes) {
    for (unsigned bad_alternative_index : bad_alternative_indexes) {
      std::vector<typename SatProblem::variable_type> clause;
      clause.reserve(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        clause.push_back(s[criterion_index][good_alternative_index][bad_alternative_index]);
      }
      sat.add_clause(clause);
    }
  }

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException();
  }

  std::vector<Model::Boundary> boundaries;

  // No loop on categories because there are only two, so only one boundary
  {
    std::vector<float> profile(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      bool found = false;
      // @todo Replace next loop with a binary search
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        if ((*solution)[a[criterion_index][value_index]]) {
          if (value_index == 0) {
            profile[criterion_index] = unique_values[criterion_index][value_index];
          } else {
            profile[criterion_index] = (unique_values[criterion_index][value_index - 1] + unique_values[criterion_index][value_index]) / 2;
          }
          found = true;
          break;
        }
      }
      if (!found) {
        profile[criterion_index] = 1;  // @todo Use the max value for the criterion
      }
    }

    std::vector<std::vector<unsigned>> roots;
    for (unsigned good_alternative_index : good_alternative_indexes) {
      std::vector<unsigned>& root = roots.emplace_back();
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (learning_set.alternatives[good_alternative_index].profile[criterion_index] >= profile[criterion_index]) {
          root.push_back(criterion_index);
        }
      }
    }

    boundaries.emplace_back(profile, SufficientCoalitions{SufficientCoalitions::roots, criteria_count, roots});
  }

  return Model{problem, boundaries};
}

template class SatSeparationUcncsLearning<EvalmaxsatSatProblem>;
template class SatSeparationUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
