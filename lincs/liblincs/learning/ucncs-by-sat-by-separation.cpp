// Copyright 2023 Vincent Jacques

#include "ucncs-by-sat-by-separation.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

#include "exception.hpp"
#include "../sat/minisat.hpp"


namespace lincs {

template<typename SatProblem>
Model SatSeparationUcncsLearning<SatProblem>::perform() {
  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.categories.size();
  const unsigned boundaries_count = categories_count - 1;
  const unsigned alternatives_count = learning_set.alternatives.size();

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

  // Alternatives above category k
  std::vector<std::vector<unsigned>> better_alternative_indexes(categories_count);
  // Alternatives in category k or below
  std::vector<std::vector<unsigned>> worse_alternative_indexes(categories_count);
  for (unsigned category_index = 0; category_index != categories_count; ++category_index) {
    better_alternative_indexes[category_index].reserve(alternatives_count);
    worse_alternative_indexes[category_index].reserve(alternatives_count);
  }
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    for (unsigned category_index = 0; category_index != *learning_set.alternatives[alternative_index].category_index; ++category_index) {
      better_alternative_indexes[category_index].push_back(alternative_index);
    }
    for (unsigned category_index = *learning_set.alternatives[alternative_index].category_index; category_index != categories_count; ++category_index) {
      worse_alternative_indexes[category_index].push_back(alternative_index);
    }
  }
  for (unsigned category_index = 0; category_index != categories_count; ++category_index) {
    assert(better_alternative_indexes[category_index].size() + worse_alternative_indexes[category_index].size() == alternatives_count);
  }

  SatProblem sat;

  // a[i][k][x]: value x is above profile k on criterion i
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> a(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    a[criterion_index].resize(boundaries_count);
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      a[criterion_index][boundary_index].resize(unique_values[criterion_index].size());
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        a[criterion_index][boundary_index][value_index] = sat.create_variable();
      }
    }
  }

  // s[i][k][k'][g][b]: criterion i separates alternatives g and b with regards to profiles k and k'
  std::vector<std::vector<std::vector<std::vector<std::vector<typename SatProblem::variable_type>>>>> s(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    s[criterion_index].resize(boundaries_count);
    for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
      s[criterion_index][boundary_index_a].resize(boundaries_count);
      for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
        s[criterion_index][boundary_index_a][boundary_index_b].resize(alternatives_count);
        for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
          s[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index].resize(alternatives_count);
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            s[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index] = sat.create_variable();
          }
        }
      }
    }
  }

  sat.mark_all_variables_created();

  // P'1
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size() - 1; ++value_index) {
        sat.add_clause({
          -a[criterion_index][boundary_index][value_index],
          a[criterion_index][boundary_index][value_index + 1],
        });
      }
    }
  }
  // P'2
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
      for (unsigned boundary_index_a = 1; boundary_index_a != boundaries_count; ++boundary_index_a) {
        sat.add_clause({
          -a[criterion_index][boundary_index_a][value_index],
          a[criterion_index][boundary_index_a - 1][value_index],
        });
      }
    }
  }

  // P'C3
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
      for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
        const unsigned bad_value_index = std::distance(
          unique_values[criterion_index].begin(),
          std::lower_bound(
            unique_values[criterion_index].begin(),
            unique_values[criterion_index].end(),
            learning_set.alternatives[bad_alternative_index].profile[criterion_index]
          )
        );
        for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
          for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
            sat.add_clause({
              -s[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              -a[criterion_index][boundary_index_a][bad_value_index],
            });
          }
        }
      }
    }
  }

  // P'C4
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        const unsigned good_value_index = std::distance(
          unique_values[criterion_index].begin(),
          std::lower_bound(
            unique_values[criterion_index].begin(),
            unique_values[criterion_index].end(),
            learning_set.alternatives[good_alternative_index].profile[criterion_index]
          )
        );
        for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            sat.add_clause({
              -s[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              a[criterion_index][boundary_index_b][good_value_index],
            });
          }
        }
      }
    }
  }

  // P'C5
  for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
    for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
          std::vector<typename SatProblem::variable_type> clause;
          clause.reserve(criteria_count);
          for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
            clause.push_back(s[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index]);
          }
          sat.add_clause(clause);
        }
      }
    }
  }

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException();
  }

  std::vector<std::vector<float>> profiles(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<float>& profile = profiles[boundary_index];
    profile.resize(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      bool found = false;
      // @todo Replace next loop with a binary search
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        if ((*solution)[a[criterion_index][boundary_index][value_index]]) {
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
  }

  std::set<std::vector<unsigned>> roots_;
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    for (unsigned good_alternative_index : better_alternative_indexes[boundary_index]) {
      std::vector<unsigned> root;
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (learning_set.alternatives[good_alternative_index].profile[criterion_index] >= profiles[boundary_index][criterion_index]) {
          root.push_back(criterion_index);
        }
      }
      roots_.insert(root);
    }
  }
  // @todo Reduce to actual roots?
  std::vector<std::vector<unsigned>> roots(roots_.begin(), roots_.end());

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    boundaries.emplace_back(profiles[boundary_index], SufficientCoalitions{SufficientCoalitions::roots, criteria_count, roots});
  }

  return Model{problem, boundaries};
}

template class SatSeparationUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
