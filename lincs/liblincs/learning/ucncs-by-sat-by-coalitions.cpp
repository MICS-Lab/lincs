// Copyright 2023-2024 Vincent Jacques

#include "ucncs-by-sat-by-coalitions.hpp"

#include <algorithm>
#include <map>
#include <type_traits>

#include "../chrones.hpp"
#include "../classification.hpp"
#include "../sat/minisat.hpp"
#include "exception.hpp"


namespace lincs {

// @todo(Project management, later) Factorize common parts of all SAT approaches
// For example:
// - 'decode' is almost identical between the 'by separation' approaches
// - 'decode' is identical between the 'by coalitions' approaches
// - 'add_structural_constraints' are identical between the 'by coalitions' approaches
// - 'add_structural_constraints' are identical between the 'by separation' approaches
// - 'add_structural_constraints' have a common part between all approaches
// - 'add_learning_set_constraints' have a common part between the 'by separation' approaches
// But:
// - 'add_learning_set_constraints' are very similar, but fundamentally different between 'by coalitions' approaches

template<typename V>
std::vector<V> implies(V a, V b) {
  // "A => B" <=> "-A or B"
  return {-a, b};
}

template<typename SatProblem>
Model SatCoalitionsUcncsLearning<SatProblem>::perform() {
  CHRONE();

  create_all_coalitions();
  create_variables();
  add_structural_constraints();
  add_learning_set_constraints();

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException("SatCoalitions failed to find a solution.");
  }

  return decode(*solution);
}

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::create_all_coalitions() {
  CHRONE();

  all_coalitions.reserve(coalitions_count);
  for (unsigned coalition_index = 0; coalition_index != coalitions_count; ++coalition_index) {
    all_coalitions.emplace_back(learning_set.criteria_count, coalition_index);
  }
}

// This implementation was initially based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its "Definition 4.1", with the special case for Uc-NCS at the end of section 4.1.
// That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.
// This implementation was then modified to handle single-peaked criteria (see clauses C1 below).

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::create_variables() {
  CHRONE();

  // Variables "a" in the article
  accepted.resize(learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    accepted[criterion_index].resize(learning_set.categories_count);
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      accepted[criterion_index][boundary_index].resize(learning_set.values_counts[criterion_index]);
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        accepted[criterion_index][boundary_index][value_rank] = sat.create_variable();
      }
    }
  }

  // Variables "t" in the article
  sufficient.resize(coalitions_count);
  for (const Coalition& coalition : all_coalitions) {
    sufficient[coalition.to_ulong()] = sat.create_variable();
  }

  sat.mark_all_variables_created();
}

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::add_structural_constraints() {
  CHRONE();

  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    const unsigned values_count = learning_set.values_counts[criterion_index];

    if (learning_set.single_peaked[criterion_index]) {
      // In this branch, "accepted" means "inside the interval"

      if (values_count >= 3) {
        for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
          for (unsigned value_rank_a = 0; value_rank_a != values_count - 2; ++value_rank_a) {
            for (unsigned value_rank_c = value_rank_a + 2; value_rank_c != values_count; ++value_rank_c) {
              sat.add_clause({
                -accepted[criterion_index][boundary_index][value_rank_a],
                -accepted[criterion_index][boundary_index][value_rank_c],
                // These two variables are the same when value_rank_c == value_rank_a + 2, but it doesn't hurt
                accepted[criterion_index][boundary_index][value_rank_a + 1],
                accepted[criterion_index][boundary_index][value_rank_c - 1],
              });
            }
          }
        }
      }
    } else {
      // In this branch, "accepted" means "above the profile"

      // Clauses "C1" in the article
      // Values are ordered so if a value is above a profile, then values above it are also above that profile
      for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
        for (unsigned value_rank = 1; value_rank != values_count; ++value_rank) {
          sat.add_clause(implies(
            accepted[criterion_index][boundary_index][value_rank - 1],
            accepted[criterion_index][boundary_index][value_rank]
          ));
        }
      }
    }
  }

  // Clauses "C2" in the article
  // Boundaries are ordered so if a value is accepted by a boundary, then it is also accepted by less strict boundaries
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
      for (unsigned boundary_index = 1; boundary_index != learning_set.boundaries_count; ++boundary_index) {
        sat.add_clause(implies(
          accepted[criterion_index][boundary_index][value_rank],
          accepted[criterion_index][boundary_index - 1][value_rank]
        ));
      }
    }
  }

  // Clauses "C3" in the article
  // Coalitions form an upset so if a coalition is sufficient, then all coalitions that include it are sufficient too
  // @todo(Performance, later) Optimize this nested loop using the fact that a is included in b
  // Or even better, add constraints only for the transitive reduction of the inclusion relation
  // Same in "max-SAT by coalitions" approach
  for (const auto& coalition_a : all_coalitions) {
    for (const auto& coalition_b : all_coalitions) {
      if (coalition_a.is_proper_subset_of(coalition_b)) {
        sat.add_clause(implies(sufficient[coalition_a.to_ulong()], sufficient[coalition_b.to_ulong()]));
      }
    }
  }

  // No need for clauses "C4", as stated in the special case for Uc-NCS
}

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::add_learning_set_constraints() {
  CHRONE();

  // Clauses "C5" in the article
  // Alternatives are not accepted by the boundary of the category better than theirs
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    const unsigned category_index = learning_set.assignments[alternative_index];
    if (category_index == learning_set.categories_count - 1) {
      continue;
    }

    // This boundary barely doesn't accept the alternative, so it can't accept the alternative on a sufficient coalition
    const unsigned boundary_index = category_index;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename SatProblem::variable_type> clause;
      // Either the coalition is not sufficient...
      clause.push_back(-sufficient[coalition.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const unsigned value_rank = learning_set.performance_ranks[criterion_index][alternative_index];
          assert(value_rank < accepted[criterion_index][boundary_index].size());
          // ... or the alternative is not accepted on at least one necessary criterion
          clause.push_back(-accepted[criterion_index][boundary_index][value_rank]);
        }
      }
      sat.add_clause(clause);
    }
  }

  // Clauses "C6" in the article
  // Alternatives are accepted by the boundary of their category
  for (unsigned alternative_index = 0; alternative_index != learning_set.alternatives_count; ++alternative_index) {
    const unsigned category_index = learning_set.assignments[alternative_index];
    if (category_index == 0) {
      continue;
    }

    // This boundary barely accepts the alternative, so it has to accept the alternative on a sufficient coalition
    const unsigned boundary_index = category_index - 1;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename SatProblem::variable_type> clause;
      const Coalition coalition_complement = ~coalition;
      clause.push_back(sufficient[coalition_complement.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const unsigned value_rank = learning_set.performance_ranks[criterion_index][alternative_index];
          assert(value_rank < accepted[criterion_index][boundary_index].size());
          clause.push_back(accepted[criterion_index][boundary_index][value_rank]);
        }
      }
      sat.add_clause(clause);
    }
  }
}

template<typename SatProblem>
Model SatCoalitionsUcncsLearning<SatProblem>::decode(const std::vector<bool>& solution) {
  CHRONE();

  std::vector<Coalition> roots;
  for (const auto& coalition_a : all_coalitions) {
    if (solution[sufficient[coalition_a.to_ulong()]]) {
      bool coalition_a_is_root = true;
      // @todo(Performance, later) Optimize this search for actual roots; it may be something like a transitive reduction
      // Same in "max-SAT by coalitions" approach
      // Same in "SAT by separation" approach
      // Same in "max-SAT by separation" approach
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

  std::vector<PreprocessedBoundary> boundaries;
  boundaries.reserve(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>> profile_ranks(learning_set.criteria_count);
    for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
      const unsigned values_count = learning_set.values_counts[criterion_index];
      if (learning_set.single_peaked[criterion_index]) {
        // In this branch, "accepted" means "inside the interval"

        bool found = false;
        unsigned low = 0;
        unsigned high = values_count;
        for (unsigned value_rank = 0; value_rank != values_count; ++value_rank) {
          if (solution[accepted[criterion_index][boundary_index][value_rank]]) {
            if (!found) {
              low = value_rank;
            }
            found = true;
            high = value_rank;
          }
        }
        if (found) {
          profile_ranks[criterion_index] = std::make_pair(low, high);
        } else {
          // Past-the-end rank
          profile_ranks[criterion_index] = std::make_pair(values_count, values_count);
        }
      } else {
        // In this branch, "accepted" means "above the profile"

        bool found = false;
        // @todo(Performance, later) Replace next loop with a binary search
        // Same in "max-SAT by coalitions" approach
        // Same in "SAT by separation" approach
        // Same in "max-SAT by separation" approach
        for (unsigned value_rank = 0; value_rank != values_count; ++value_rank) {
          if (solution[accepted[criterion_index][boundary_index][value_rank]]) {
            profile_ranks[criterion_index] = value_rank;
            found = true;
            break;
          }
        }
        if (!found) {
          // Past-the-end rank
          profile_ranks[criterion_index] = values_count;
        }
      }
    }

    boundaries.emplace_back(profile_ranks, SufficientCoalitions(SufficientCoalitions::Roots(Internal(), roots)));
  }

  const Model model = learning_set.post_process(boundaries);
  assert(count_correctly_classified_alternatives(input_problem, model, input_learning_set) == learning_set.alternatives_count);
  return model;
}

template class SatCoalitionsUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
