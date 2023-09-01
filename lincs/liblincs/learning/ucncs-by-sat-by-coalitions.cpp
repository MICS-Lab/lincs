// Copyright 2023 Vincent Jacques

#include "ucncs-by-sat-by-coalitions.hpp"

#include <algorithm>
#include <map>
#include <type_traits>

#include "exception.hpp"
#include "../sat/minisat.hpp"


namespace lincs {

// @todo(Project management, soon-ish) Factorize common parts of all SAT approaches

template<typename V>
std::vector<V> implies(V a, V b) {
  // "A => B" <=> "-A or B"
  return {-a, b};
}

template<typename SatProblem>
Model SatCoalitionsUcncsLearning<SatProblem>::perform() {
  sort_values();
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

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::sort_values() {
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

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::create_all_coalitions() {
  all_coalitions.reserve(coalitions_count);
  for (unsigned coalition_index = 0; coalition_index != coalitions_count; ++coalition_index) {
    all_coalitions.emplace_back(criteria_count, coalition_index);
  }
}

// This implementation is based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its "Definition 4.1", with the special case for Uc-NCS at the end of section 4.1.
// That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.

template<typename SatProblem>
void SatCoalitionsUcncsLearning<SatProblem>::create_variables() {
  // Variables "a" in the article
  better.resize(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    better[criterion_index].resize(categories_count);
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      better[criterion_index][boundary_index].resize(unique_values[criterion_index].size());
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        better[criterion_index][boundary_index][value_index] = sat.create_variable();
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
  // Clauses "C1" in the article
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

  // Clauses "C2" in the article
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
  // Clauses "C5" in the article
  // Alternatives are outranked by the boundary better than them
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const auto& alternative = learning_set.alternatives[alternative_index];

    const unsigned category_index = *alternative.category_index;
    if (category_index == categories_count - 1) {
      continue;
    }

    // This boundary is *just better than* the alternative, so the alternative can't be better than it on a sufficient coalition
    const unsigned boundary_index = category_index;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename SatProblem::variable_type> clause;
      // Either the coalition is not sufficient...
      clause.push_back(-sufficient[coalition.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const Criterion& criterion = problem.criteria[criterion_index];
          const auto lb = std::lower_bound(
            unique_values[criterion_index].begin(), unique_values[criterion_index].end(),
            alternative.profile[criterion_index],
            [&criterion](float lhs, float rhs) { return criterion.strictly_better(rhs, lhs); }
          );
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = std::distance(unique_values[criterion_index].begin(), lb);
          assert(value_index < better[criterion_index][boundary_index].size());
          // ... or the alternative is worse than the profile on at least one necessary criterion
          clause.push_back(-better[criterion_index][boundary_index][value_index]);
        }
      }
      sat.add_clause(clause);
    }
  }

  // Clauses "C6" in the article
  // Alternatives outrank the boundary worse than them
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const auto& alternative = learning_set.alternatives[alternative_index];

    const unsigned category_index = *alternative.category_index;
    if (category_index == 0) {
      continue;
    }

    // This boundary is *just worse than* the alternative, so the alternative has to be better than it on a sufficient coalition
    const unsigned boundary_index = category_index - 1;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename SatProblem::variable_type> clause;
      const Coalition coalition_complement = ~coalition;
      clause.push_back(sufficient[coalition_complement.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const Criterion& criterion = problem.criteria[criterion_index];
          const auto lb = std::lower_bound(
            unique_values[criterion_index].begin(), unique_values[criterion_index].end(),
            alternative.profile[criterion_index],
            [&criterion](float lhs, float rhs) { return criterion.strictly_better(rhs, lhs); }
          );
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = std::distance(unique_values[criterion_index].begin(), lb);
          assert(value_index < better[criterion_index][boundary_index].size());
          clause.push_back(better[criterion_index][boundary_index][value_index]);
        }
      }
      sat.add_clause(clause);
    }
  }
}

template<typename SatProblem>
Model SatCoalitionsUcncsLearning<SatProblem>::decode(const std::vector<bool>& solution) {
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

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<float> profile(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      const bool is_growing = problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::growing;
      assert(is_growing || problem.criteria[criterion_index].category_correlation == Criterion::CategoryCorrelation::decreasing);
      const float best_value = is_growing ? problem.criteria[criterion_index].max_value : problem.criteria[criterion_index].min_value;
      const float worst_value = is_growing ? problem.criteria[criterion_index].min_value : problem.criteria[criterion_index].max_value;

      bool found = false;
      // @todo(Performance, later) Replace next loop with a binary search
      // Same in "max-SAT by coalitions" approach
      // Same in "SAT by separation" approach
      // Same in "max-SAT by separation" approach
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

    boundaries.emplace_back(profile, SufficientCoalitions{SufficientCoalitions::roots, roots});
  }

  return Model{problem, boundaries};
}

template class SatCoalitionsUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
