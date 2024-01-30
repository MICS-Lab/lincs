// Copyright 2023-2024 Vincent Jacques

#include "ucncs-by-max-sat-by-separation.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

#include "../chrones.hpp"
#include "../sat/eval-max-sat.hpp"
#include "exception.hpp"


namespace lincs {

template<typename V>
std::vector<V> implies(V a, V b) {
  // "A => B" <=> "-A or B"
  return {-a, b};
}

template<typename MaxSatProblem>
Model MaxSatSeparationUcncsLearning<MaxSatProblem>::perform() {
  CHRONE();

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
void MaxSatSeparationUcncsLearning<MaxSatProblem>::partition_alternatives() {
  CHRONE();

  better_alternative_indexes.resize(learning_set.categories_count);
  worse_alternative_indexes.resize(learning_set.categories_count);
  for (unsigned category_index = 0; category_index != learning_set.categories_count; ++category_index) {
    better_alternative_indexes[category_index].reserve(learning_set.alternatives_count);
    worse_alternative_indexes[category_index].reserve(learning_set.alternatives_count);
  }
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    for (unsigned category_index = 0; category_index != learning_set.assignments[alternative_index]; ++category_index) {
      better_alternative_indexes[category_index].push_back(alternative_index);
    }
    for (unsigned category_index = learning_set.assignments[alternative_index]; category_index != learning_set.categories_count; ++category_index) {
      worse_alternative_indexes[category_index].push_back(alternative_index);
    }
  }
  for (unsigned category_index = 0; category_index != learning_set.categories_count; ++category_index) {
    assert(better_alternative_indexes[category_index].size() + worse_alternative_indexes[category_index].size() == learning_set.alternatives_count);
  }
}

// This implementation is based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its section B2. That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::create_variables() {
  CHRONE();

  // Variables "a" in the article
  better.resize(learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    better[criterion_index].resize(learning_set.boundaries_count);
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      better[criterion_index][boundary_index].resize(learning_set.values_counts[criterion_index]);
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        better[criterion_index][boundary_index][value_rank] = sat.create_variable();
      }
    }
  }

  // Variables "s" in the article
  separates.resize(learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    separates[criterion_index].resize(learning_set.boundaries_count);
    for (unsigned boundary_index_a = 0; boundary_index_a != learning_set.boundaries_count; ++boundary_index_a) {
      separates[criterion_index][boundary_index_a].resize(learning_set.boundaries_count);
      for (unsigned boundary_index_b = 0; boundary_index_b != learning_set.boundaries_count; ++boundary_index_b) {
        separates[criterion_index][boundary_index_a][boundary_index_b].resize(learning_set.alternatives_count);
        for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
          separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index].resize(learning_set.alternatives_count);
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index] = sat.create_variable();
          }
        }
      }
    }
  }

  // Variables "z" in the article
  correct.resize(learning_set.alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    correct[alternative_index] = sat.create_variable();
  }

  // Variables "y" in the article
  proper.resize(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    proper[boundary_index].resize(learning_set.alternatives_count);
    for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
      proper[boundary_index][alternative_index] = sat.create_variable();
    }
  }

  sat.mark_all_variables_created();
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::add_structural_constraints() {
  CHRONE();

  // Clauses "P'1" in the article
  // Values are ordered so if a value is better than a profile, then values better than it are also better than that profile
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      for (unsigned value_rank = 1; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        sat.add_clause(implies(
          better[criterion_index][boundary_index][value_rank - 1],
          better[criterion_index][boundary_index][value_rank]
        ));
      }
    }
  }

  // Clauses "P'2" in the article
  // Profiles are ordered so if a value is better than a profile, then it is also better than lower profiles
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
      for (unsigned boundary_index = 1; boundary_index != learning_set.boundaries_count; ++boundary_index) {
        sat.add_clause(implies(
          better[criterion_index][boundary_index][value_rank],
          better[criterion_index][boundary_index - 1][value_rank]
        ));
      }
    }
  }
}

template<typename MaxSatProblem>
void MaxSatSeparationUcncsLearning<MaxSatProblem>::add_learning_set_constraints() {
  CHRONE();

  // Clauses "P'C3" in the article
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    for (unsigned boundary_index_a = 0; boundary_index_a != learning_set.boundaries_count; ++boundary_index_a) {
      for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
        const unsigned bad_value_rank = learning_set.performance_ranks[criterion_index][bad_alternative_index];
        for (unsigned boundary_index_b = 0; boundary_index_b != learning_set.boundaries_count; ++boundary_index_b) {
          for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
            sat.add_clause(implies(
              separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              -better[criterion_index][boundary_index_a][bad_value_rank]
            ));
          }
        }
      }
    }
  }

  // Clauses "P'C4" in the article
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    for (unsigned boundary_index_b = 0; boundary_index_b != learning_set.boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        const unsigned good_value_rank = learning_set.performance_ranks[criterion_index][good_alternative_index];
        for (unsigned boundary_index_a = 0; boundary_index_a != learning_set.boundaries_count; ++boundary_index_a) {
          for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
            sat.add_clause(implies(
              separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index],
              better[criterion_index][boundary_index_b][good_value_rank]
            ));
          }
        }
      }
    }
  }

  // Clauses "P'C5~" in the article
  for (unsigned boundary_index_a = 0; boundary_index_a != learning_set.boundaries_count; ++boundary_index_a) {
    for (unsigned boundary_index_b = 0; boundary_index_b != learning_set.boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
          std::vector<typename MaxSatProblem::variable_type> clause;
          clause.reserve(learning_set.criteria_count + 2);
          for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
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
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      sat.add_clause(implies(
        correct[alternative_index],
        proper[boundary_index][alternative_index]
      ));
    }
  }

  // Maximize the number of alternatives classified correctly
  // Clauses "goal" in the article
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    sat.add_weighted_clause({correct[alternative_index]}, goal_weight);
  }
  // Clauses "subgoals" in the article
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      sat.add_weighted_clause({proper[boundary_index][alternative_index]}, subgoal_weight);
    }
  }
}

template<typename MaxSatProblem>
Model MaxSatSeparationUcncsLearning<MaxSatProblem>::decode(const std::vector<bool>& solution) {
  CHRONE();

  std::vector<std::vector<unsigned>> profile_ranks(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    std::vector<unsigned>& ranks = profile_ranks[boundary_index];
    ranks.resize(learning_set.criteria_count);
    for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
      bool found = false;
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        if (solution[better[criterion_index][boundary_index][value_rank]]) {
          ranks[criterion_index] = value_rank;
          found = true;
          break;
        }
      }
      if (!found) {
        // Past-the-end rank
        ranks[criterion_index] = learning_set.values_counts[criterion_index];
      }
    }
  }

  std::set<boost::dynamic_bitset<>> sufficient_coalitions;
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    for (unsigned good_alternative_index : better_alternative_indexes[boundary_index]) {
      if (!solution[correct[good_alternative_index]]) {
        // Alternative is not correctly classified, it does not participate
        continue;
      }

      boost::dynamic_bitset<> coalition(learning_set.criteria_count);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        const bool is_better = learning_set.performance_ranks[criterion_index][good_alternative_index] >= profile_ranks[boundary_index][criterion_index];
        if (is_better) {
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

  std::vector<PreProcessedBoundary> boundaries;
  boundaries.reserve(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    boundaries.emplace_back(profile_ranks[boundary_index], SufficientCoalitions(SufficientCoalitions::Roots(Internal(), roots)));
  }

  return learning_set.post_process(boundaries);
}

template class MaxSatSeparationUcncsLearning<EvalmaxsatMaxSatProblem>;

}  // namespace lincs
