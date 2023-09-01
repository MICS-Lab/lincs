// Copyright 2023 Vincent Jacques

#include "ucncs-by-max-sat-by-separation.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

#include "exception.hpp"
#include "../sat/eval-max-sat.hpp"


namespace lincs {

template<typename V>
std::vector<V> implies(V a, V b) {
  // "A => B" <=> "-A or B"
  return {-a, b};
}

template<typename MaxSatProblem>
Model MaxSatSeparationUcncsLearning<MaxSatProblem>::perform() {
  sort_values();
  partition_alternatives();
  create_variables();
  add_structural_constraints();
  add_learning_set_constraints();

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException();
  }

  return decode(*solution);
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::sort_values() {
  unique_values.resize(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    unique_values[criterion_index].reserve(alternatives_count);
  }
  for (const auto& alternative : learning_set.alternatives) {
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      unique_values[criterion_index].push_back(alternative.profile[criterion_index]);
    }
  }

  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    std::vector<float>& v = unique_values[criterion_index];
    std::sort(v.begin(), v.end(), [&criterion](float lhs, float rhs) { return criterion.strictly_better(rhs, lhs); });
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::partition_alternatives() {
  better_alternative_indexes.resize(categories_count);
  worse_alternative_indexes.resize(categories_count);
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
}

// This implementation is based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its section B2. That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::create_variables() {
  // Variables "a" in the article
  better.resize(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    better[criterion_index].resize(boundaries_count);
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      better[criterion_index][boundary_index].resize(unique_values[criterion_index].size());
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        better[criterion_index][boundary_index][value_index] = sat.create_variable();
      }
    }
  }

  // Variables "s" in the article
  separates.resize(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    separates[criterion_index].resize(boundaries_count);
    for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
      separates[criterion_index][boundary_index_a].resize(boundaries_count);
      for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
        separates[criterion_index][boundary_index_a][boundary_index_b].resize(alternatives_count);
        for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
          separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index].resize(alternatives_count);
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index] = sat.create_variable();
          }
        }
      }
    }
  }

  // Variables "z" in the article
  correct.resize(alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    correct[alternative_index] = sat.create_variable();
  }

  // Variables "y" in the article
  proper.resize(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    proper[boundary_index].reserve(alternatives_count);
    for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
      proper[boundary_index][alternative_index] = sat.create_variable();
    }
  }

  sat.mark_all_variables_created();
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::add_structural_constraints() {
  // Clauses "P'1" in the article
  // Values are ordered so if a value is better than a profile, then values better than it are also better than that profile
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      for (unsigned value_index = 1; value_index != unique_values[criterion_index].size(); ++value_index) {
        sat.add_clause(implies(
          better[criterion_index][boundary_index][value_index - 1],
          better[criterion_index][boundary_index][value_index]
        ));
      }
    }
  }

  // Clauses "P'2" in the article
  // Profiles are ordered so if a value is better than a profile, then it is also better than lower profiles
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
      for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
        sat.add_clause(implies(
          better[criterion_index][boundary_index][value_index],
          better[criterion_index][boundary_index - 1][value_index]
        ));
      }
    }
  }
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::add_learning_set_constraints() {
  // Clauses "P'C3" in the article
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
      for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
        const auto lb = std::lower_bound(
          unique_values[criterion_index].begin(),
          unique_values[criterion_index].end(),
          learning_set.alternatives[bad_alternative_index].profile[criterion_index],
          [&criterion](float lhs, float rhs) { return criterion.strictly_better(rhs, lhs); }
        );
        assert(lb != unique_values[criterion_index].end());
        const unsigned bad_value_index = std::distance(unique_values[criterion_index].begin(), lb);
        for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
          for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
            sat.add_clause(implies(
              separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              -better[criterion_index][boundary_index_a][bad_value_index]
            ));
          }
        }
      }
    }
  }

  // Clauses "P'C4" in the article
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    const Criterion& criterion = problem.criteria[criterion_index];
    for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        const auto lb = std::lower_bound(
          unique_values[criterion_index].begin(),
          unique_values[criterion_index].end(),
          learning_set.alternatives[good_alternative_index].profile[criterion_index],
          [&criterion](float lhs, float rhs) { return criterion.strictly_better(rhs, lhs); }
        );
        assert(lb != unique_values[criterion_index].end());
        const unsigned good_value_index = std::distance(unique_values[criterion_index].begin(), lb);
        for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            sat.add_clause(implies(
              separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              better[criterion_index][boundary_index_b][good_value_index]
            ));
          }
        }
      }
    }
  }

  // Clauses "P'C5~" in the article
  for (unsigned boundary_index_a = 0; boundary_index_a != boundaries_count; ++boundary_index_a) {
    for (unsigned boundary_index_b = 0; boundary_index_b != boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
          std::vector<typename MaxSatProblem::variable_type> clause;
          clause.reserve(criteria_count + 2);
          for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
            clause.push_back(separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index]);
          }
          clause.push_back(-proper[boundary_index_a][bad_alternative_index]);
          clause.push_back(-proper[boundary_index_b][good_alternative_index]);
          sat.add_clause(clause);
        }
      }
    }
  }

  // Clauses "P'yz~" in the article
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      sat.add_clause(implies(
        correct[alternative_index],
        proper[boundary_index][alternative_index]
      ));
    }
  }

  // Maximize the number of alternatives classified correctly
  // Clauses "goal" in the article
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    sat.add_weighted_clause({correct[alternative_index]}, goal_weight);
  }
  // Clauses "subgoals" in the article
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      sat.add_weighted_clause({proper[boundary_index][alternative_index]}, subgoal_weight);
    }
  }
}

template<typename MaxSatProblem>
Model MaxSatSeparationUcncsLearning<MaxSatProblem>::decode(const std::vector<bool>& solution) {
  std::vector<std::vector<float>> profiles(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<float>& profile = profiles[boundary_index];
    profile.resize(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const bool is_growing = problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::growing;
      assert(is_growing || problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::decreasing);
      const float best_value = is_growing ? problem.criteria[criterion_index].max_value : problem.criteria[criterion_index].min_value;
      const float worst_value = is_growing ? problem.criteria[criterion_index].min_value : problem.criteria[criterion_index].max_value;

      bool found = false;
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        if (solution[better[criterion_index][boundary_index][value_index]]) {
          if (value_index == 0) {
            profile[criterion_index] = worst_value;
          } else {
            profile[criterion_index] = (unique_values[criterion_index][value_index - 1] + unique_values[criterion_index][value_index]) / 2;
          }
          found = true;
          break;
        }
      }
      if (!found) {
        profile[criterion_index] = best_value;
      }
    }
  }

  std::set<boost::dynamic_bitset<>> sufficient_coalitions;
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    for (unsigned good_alternative_index : better_alternative_indexes[boundary_index]) {
      if (!solution[correct[good_alternative_index]]) {
        // Alternative is not correctly classified, it does not participate
        continue;
      }

      boost::dynamic_bitset<> coalition(criteria_count);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (problem.criteria[criterion_index].better_or_equal(learning_set.alternatives[good_alternative_index].profile[criterion_index], profiles[boundary_index][criterion_index])) {
          coalition.set(criterion_index);
        }
      }
      sufficient_coalitions.insert(coalition);
    }
  }
  std::vector<boost::dynamic_bitset<>> roots;
  for (const auto& coalition_a : sufficient_coalitions) {
    bool coalition_a_is_root = true;
    for (const auto& coalition_b : sufficient_coalitions) {
      if (coalition_b.is_proper_subset_of(coalition_a)) {
        coalition_a_is_root = false;
        break;
      }
    }
    if (coalition_a_is_root) {
      roots.push_back(coalition_a);
    }
  }

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    boundaries.emplace_back(profiles[boundary_index], SufficientCoalitions{SufficientCoalitions::roots, roots});
  }

  return Model{problem, boundaries};
}

template class MaxSatSeparationUcncsLearning<EvalmaxsatMaxSatProblem>;

}  // namespace lincs
