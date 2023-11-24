// Copyright 2023 Vincent Jacques

#include "ucncs-by-max-sat-by-coalitions.hpp"

#include <algorithm>
#include <map>
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
Model MaxSatCoalitionsUcncsLearning<MaxSatProblem>::perform() {
  CHRONE();

  create_all_coalitions();
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
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::create_all_coalitions() {
  CHRONE();

  all_coalitions.reserve(coalitions_count);
  for (unsigned coalition_index = 0; coalition_index != coalitions_count; ++coalition_index) {
    all_coalitions.emplace_back(learning_set.criteria_count, coalition_index);
  }
}

// This implementation is based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its section 5.1, with the special case for Uc-NCS at the end of the section.
// That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::create_variables() {
  CHRONE();

  // Variables "a" in the article
  better.resize(learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    better[criterion_index].resize(learning_set.categories_count);
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      better[criterion_index][boundary_index].resize(learning_set.values_counts[criterion_index]);
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        better[criterion_index][boundary_index][value_rank] = sat.create_variable();
      }
    }
  }

  // Variables "t" in the article
  sufficient.resize(coalitions_count);
  for (const Coalition& coalition : all_coalitions) {
    sufficient[coalition.to_ulong()] = sat.create_variable();
  }

  // Variables "z" in the article
  correct.resize(learning_set.alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    correct[alternative_index] = sat.create_variable();
  }

  sat.mark_all_variables_created();
}

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::add_structural_constraints() {
  CHRONE();

  // Clauses "C1" in the article
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

  // Clauses "C2" in the article
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

  // Clauses "C3" in the article
  // Coalitions form an upset so if a coalition is sufficient, then all coalitions that include it are sufficient too
  for (const auto& coalition_a : all_coalitions) {
    for (const auto& coalition_b : all_coalitions) {
      if (coalition_a.is_proper_subset_of(coalition_b)) {
        sat.add_clause(implies(sufficient[coalition_a.to_ulong()], sufficient[coalition_b.to_ulong()]));
      }
    }
  }

  // No need for clauses "C4"
}

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::add_learning_set_constraints() {
  CHRONE();

  // Clauses "C5~" in the article
  // Alternatives are outranked by the boundary better than them
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    const unsigned category_index = learning_set.assignments[alternative_index];
    if (category_index == learning_set.categories_count - 1) {
      continue;
    }

    // This boundary is *just better than* the alternative, so the alternative can't be better than it on a sufficient coalition
    const unsigned boundary_index = category_index;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename MaxSatProblem::variable_type> clause;
      // Either the coalition is not sufficient...
      clause.push_back(-sufficient[coalition.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const unsigned value_rank = learning_set.performance_ranks[criterion_index][alternative_index];
          assert(value_rank < better[criterion_index][boundary_index].size());
          // ... or the alternative is worse than the profile on at least one necessary criterion
          clause.push_back(-better[criterion_index][boundary_index][value_rank]);
        }
      }
      // ... or it's not correctly classified
      clause.push_back(-correct[alternative_index]);
      sat.add_clause(clause);
    }
  }

  // Clauses "C6~" in the article
  // Alternatives outrank the boundary worse than them
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    const unsigned category_index = learning_set.assignments[alternative_index];
    if (category_index == 0) {
      continue;
    }

    // This boundary is *just worse than* the alternative, so the alternative has to be better than it on a sufficient coalition
    const unsigned boundary_index = category_index - 1;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename MaxSatProblem::variable_type> clause;
      const Coalition coalition_complement = ~coalition;
      clause.push_back(sufficient[coalition_complement.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const unsigned value_rank = learning_set.performance_ranks[criterion_index][alternative_index];
          assert(value_rank < better[criterion_index][boundary_index].size());
          clause.push_back(better[criterion_index][boundary_index][value_rank]);
        }
      }
      clause.push_back(-correct[alternative_index]);
      sat.add_clause(clause);
    }
  }

  // Clauses "goal" in the article
  // Maximize the number of alternatives classified correctly
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    sat.add_weighted_clause({correct[alternative_index]}, goal_weight);
  }
}

template<typename MaxSatProblem>
Model MaxSatCoalitionsUcncsLearning<MaxSatProblem>::decode(const std::vector<bool>& solution) {
  CHRONE();

  std::vector<Coalition> roots;
  for (const auto& coalition_a : all_coalitions) {
    if (solution[sufficient[coalition_a.to_ulong()]]) {
      bool coalition_a_is_root = true;
      for (const auto& coalition_b : all_coalitions) {
        if (solution[sufficient[coalition_b.to_ulong()]]) {
          if (coalition_b.is_proper_subset_of(coalition_a)) {
            coalition_a_is_root = false;
            break;
          }
        }
      }
      if (coalition_a_is_root) {
        roots.push_back(coalition_a);
      }
    }
  }

  std::vector<PreProcessedBoundary> boundaries;
  boundaries.reserve(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    std::vector<unsigned> profile_ranks(learning_set.criteria_count);
    for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
      bool found = false;
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        if (solution[better[criterion_index][boundary_index][value_rank]]) {
          profile_ranks[criterion_index] = value_rank;
          found = true;
          break;
        }
      }
      if (!found) {
        // Past-the-end rank
        profile_ranks[criterion_index] = learning_set.values_counts[criterion_index];
      }
    }

    boundaries.emplace_back(profile_ranks, SufficientCoalitions(SufficientCoalitions::Roots(roots)));
  }

  return learning_set.post_process(boundaries);
}

template class MaxSatCoalitionsUcncsLearning<EvalmaxsatMaxSatProblem>;

}  // namespace lincs
