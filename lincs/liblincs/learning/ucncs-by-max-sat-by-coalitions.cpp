// Copyright 2023 Vincent Jacques

#include "ucncs-by-max-sat-by-coalitions.hpp"

#include <algorithm>
#include <map>
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
Model MaxSatCoalitionsUcncsLearning<MaxSatProblem>::perform() {
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

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::sort_values() {
  unique_values.resize(criteria_count);
  for (unsigned i = 0; i != criteria_count; ++i) {
    unique_values[i].reserve(alternatives_count);
  }
  for (const auto& alternative : learning_set.alternatives) {
    for (unsigned i = 0; i != criteria_count; ++i) {
      unique_values[i].push_back(alternative.profile[i]);
    }
  }

  for (auto& v : unique_values) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
  }
}

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::create_all_coalitions() {
  all_coalitions.reserve(coalitions_count);
  for (unsigned coalition_index = 0; coalition_index != coalitions_count; ++coalition_index) {
    all_coalitions.emplace_back(criteria_count, coalition_index);
  }
}

// This implementation is based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its section 5.1, with the special case for Uc-NCS at the end of the section.

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::create_variables() {
  // Variables "a" in the article
  above.resize(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    above[criterion_index].resize(categories_count);
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      above[criterion_index][boundary_index].resize(unique_values[criterion_index].size());
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        above[criterion_index][boundary_index][value_index] = sat.create_variable();
      }
    }
  }

  // Variables "t" in the article
  sufficient.resize(coalitions_count);
  for (const Coalition& coalition : all_coalitions) {
    sufficient[coalition.to_ulong()] = sat.create_variable();
  }

  // Variables "z" in the article
  correct.resize(alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    correct[alternative_index] = sat.create_variable();
  }

  sat.mark_all_variables_created();
}

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::add_structural_constraints() {
  // Clauses "C1" in the article
  // Values are ordered so if a value is above a profile, then values above it are also above that profile
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      for (unsigned value_index = 1; value_index != unique_values[criterion_index].size(); ++value_index) {
        sat.add_clause(implies(
          above[criterion_index][boundary_index][value_index - 1],
          above[criterion_index][boundary_index][value_index]
        ));
      }
    }
  }

  // Clauses "C2" in the article
  // Profiles are ordered so if a value is above a profile, then it is also above lower profiles
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
      for (unsigned boundary_index = 1; boundary_index != boundaries_count; ++boundary_index) {
        sat.add_clause(implies(
          above[criterion_index][boundary_index][value_index],
          above[criterion_index][boundary_index - 1][value_index]
        ));
      }
    }
  }

  // Clauses "C3" in the article
  // Coalitions form an upset so if a coalition is sufficient, then all coalitions that include it are sufficient too
  for (const Coalition& coalition_a : all_coalitions) {
    for (const Coalition& coalition_b : all_coalitions) {
      if (coalition_a.is_proper_subset_of(coalition_b)) {
        sat.add_clause(implies(sufficient[coalition_a.to_ulong()], sufficient[coalition_b.to_ulong()]));
      }
    }
  }

  // No need for clauses "C4"
}

template<typename MaxSatProblem>
void MaxSatCoalitionsUcncsLearning<MaxSatProblem>::add_learning_set_constraints() {
  // Clauses "C5~" in the article
  // Alternatives are outranked by the boundary above them
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const auto& alternative = learning_set.alternatives[alternative_index];

    const unsigned category_index = *alternative.category_index;
    if (category_index == categories_count - 1) {
      continue;
    }

    // This boundary is *just above* the alternative, so the alternative can't be above it on a sufficient coalition
    const unsigned boundary_index = category_index;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename MaxSatProblem::variable_type> clause;
      // Either the coalition is not sufficient...
      clause.push_back(-sufficient[coalition.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const auto lb = std::lower_bound(
            unique_values[criterion_index].begin(), unique_values[criterion_index].end(),
            alternative.profile[criterion_index]
          );
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = lb - unique_values[criterion_index].begin();
          assert(value_index < above[criterion_index][boundary_index].size());
          // ... or the alternative is below the profile on at least one necessary criterion
          clause.push_back(-above[criterion_index][boundary_index][value_index]);
        }
      }
      // ... or it's not correctly classified
      clause.push_back(-correct[alternative_index]);
      sat.add_clause(clause);
    }
  }

  // Clauses "C6~" in the article
  // Alternatives outrank the boundary below them
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const auto& alternative = learning_set.alternatives[alternative_index];

    const unsigned category_index = *alternative.category_index;
    if (category_index == 0) {
      continue;
    }

    // This boundary is *just below* the alternative, so the alternative has to be above it on a sufficient coalition
    const unsigned boundary_index = category_index - 1;

    for (const Coalition& coalition : all_coalitions) {
      std::vector<typename MaxSatProblem::variable_type> clause;
      const Coalition coalition_complement = ~coalition;
      clause.push_back(sufficient[coalition_complement.to_ulong()]);
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        if (coalition[criterion_index]) {
          const auto lb = std::lower_bound(
            unique_values[criterion_index].begin(), unique_values[criterion_index].end(),
            alternative.profile[criterion_index]
          );
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = lb - unique_values[criterion_index].begin();
          assert(value_index < above[criterion_index][boundary_index].size());
          clause.push_back(above[criterion_index][boundary_index][value_index]);
        }
      }
      clause.push_back(-correct[alternative_index]);
      sat.add_clause(clause);
    }
  }

  // Clauses "goal" in the article
  // Maximize the number of alternatives classified correctly
  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    sat.add_weighted_clause({correct[alternative_index]}, goal_weight);
  }
}

template<typename MaxSatProblem>
Model MaxSatCoalitionsUcncsLearning<MaxSatProblem>::decode(const std::vector<bool>& solution) {
  std::vector<boost::dynamic_bitset<>> roots;
  for (const Coalition& coalition_a : all_coalitions) {
    if (solution[sufficient[coalition_a.to_ulong()]]) {
      bool coalition_a_is_root = true;
      for (const Coalition& coalition_b : all_coalitions) {
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
      bool found = false;
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        if (solution[above[criterion_index][boundary_index][value_index]]) {
          if (value_index == 0) {
            profile[criterion_index] = problem.criteria[criterion_index].min_value;
          } else {
            profile[criterion_index] = (unique_values[criterion_index][value_index - 1] + unique_values[criterion_index][value_index]) / 2;
          }
          found = true;
          break;
        }
      }
      if (!found) {
        profile[criterion_index] = problem.criteria[criterion_index].max_value;
      }
    }

    boundaries.emplace_back(profile, SufficientCoalitions{SufficientCoalitions::roots, roots});
  }

  return Model{problem, boundaries};
}

template class MaxSatCoalitionsUcncsLearning<EvalmaxsatMaxSatProblem>;

}  // namespace lincs