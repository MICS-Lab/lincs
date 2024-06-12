// Copyright 2023-2024 Vincent Jacques

#include "ucncs-by-sat-by-separation.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <type_traits>

#include "../chrones.hpp"
#include "../classification.hpp"
#include "../sat/minisat.hpp"
#include "exception.hpp"


namespace lincs {

template<typename V>
std::vector<V> implies(V a, V b) {
  // "A => B" <=> "-A or B"
  return {-a, b};
}

template<typename SatProblem>
Model SatSeparationUcncsLearning<SatProblem>::perform() {
  CHRONE();

  partition_alternatives();
  create_variables();
  add_structural_constraints();
  add_learning_set_constraints();

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException("SatSeparation failed to find a solution.");
  }

  return decode(*solution);
}

template<typename SatProblem>
void SatSeparationUcncsLearning<SatProblem>::partition_alternatives() {
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

// This implementation was initially based on https://www.sciencedirect.com/science/article/abs/pii/S0377221721006858,
// specifically its "Definition A.2", which references its "Definition 4.4".
// These definitions are based on its "Theorem 4.2".
// That article is a summary of this thesis: https://www.theses.fr/2022UPAST071.
// This implementation was then modified to handle single-peaked criteria (see clauses P'1 below).
// See more comments in 'ucncs-by-sat-by-coalitions.cpp' and 'ucncs-by-sat-by-separation.cpp'.

template<typename SatProblem>
void SatSeparationUcncsLearning<SatProblem>::create_variables() {
  CHRONE();

  // Variables "a" in the article
  accepted.resize(learning_set.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    accepted[criterion_index].resize(learning_set.boundaries_count);
    for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
      accepted[criterion_index][boundary_index].resize(learning_set.values_counts[criterion_index]);
      for (unsigned value_rank = 0; value_rank != learning_set.values_counts[criterion_index]; ++value_rank) {
        accepted[criterion_index][boundary_index][value_rank] = sat.create_variable();
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

  sat.mark_all_variables_created();
}

template<typename SatProblem>
void SatSeparationUcncsLearning<SatProblem>::add_structural_constraints() {
  CHRONE();

  for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
    const unsigned values_count = learning_set.values_counts[criterion_index];

    if (learning_set.single_peaked[criterion_index]) {
      if (values_count >= 3) {
        for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
          for (unsigned value_rank_a = 0; value_rank_a != values_count - 2; ++value_rank_a) {
            for (unsigned value_rank_c = value_rank_a + 2; value_rank_c != values_count; ++value_rank_c) {
              sat.add_clause({
                -accepted[criterion_index][boundary_index][value_rank_a],
                -accepted[criterion_index][boundary_index][value_rank_c],
                accepted[criterion_index][boundary_index][value_rank_a + 1],
                accepted[criterion_index][boundary_index][value_rank_c - 1],
              });
            }
          }
        }
      }
    } else {
      // Clauses "P'1" in the article
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

  // Clauses "P'2" in the article
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
}

template<typename SatProblem>
void SatSeparationUcncsLearning<SatProblem>::add_learning_set_constraints() {
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
              -accepted[criterion_index][boundary_index_a][bad_value_rank]
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
              accepted[criterion_index][boundary_index_b][good_value_rank]
            ));
          }
        }
      }
    }
  }

  // Clauses "P'C5" in the article
  for (unsigned boundary_index_a = 0; boundary_index_a != learning_set.boundaries_count; ++boundary_index_a) {
    for (unsigned boundary_index_b = 0; boundary_index_b != learning_set.boundaries_count; ++boundary_index_b) {
      for (unsigned good_alternative_index : better_alternative_indexes[boundary_index_b]) {
        for (unsigned bad_alternative_index : worse_alternative_indexes[boundary_index_a]) {
          std::vector<typename SatProblem::variable_type> clause;
          clause.reserve(learning_set.criteria_count + 1);
          for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
            clause.push_back(separates[criterion_index][boundary_index_a][boundary_index_b][good_alternative_index][bad_alternative_index]);
          }
          sat.add_clause(clause);
        }
      }
    }
  }
}

template<typename SatProblem>
Model SatSeparationUcncsLearning<SatProblem>::decode(const std::vector<bool>& solution) {
  CHRONE();

  std::vector<std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>>> profile_ranks(learning_set.boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>>& ranks = profile_ranks[boundary_index];
    ranks.resize(learning_set.criteria_count);
    for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
      const unsigned values_count = learning_set.values_counts[criterion_index];
      if (learning_set.single_peaked[criterion_index]) {
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
          ranks[criterion_index] = std::make_pair(low, high);
        } else {
          ranks[criterion_index] = std::make_pair(values_count, values_count);
        }
      } else {
        bool found = false;
        for (unsigned value_rank = 0; value_rank != values_count; ++value_rank) {
          if (solution[accepted[criterion_index][boundary_index][value_rank]]) {
            ranks[criterion_index] = value_rank;
            found = true;
            break;
          }
        }
        if (!found) {
          ranks[criterion_index] = values_count;
        }
      }
    }
  }

  std::set<boost::dynamic_bitset<>> sufficient_coalitions;
  for (unsigned boundary_index = 0; boundary_index != learning_set.boundaries_count; ++boundary_index) {
    for (unsigned good_alternative_index : better_alternative_indexes[boundary_index]) {
      boost::dynamic_bitset<> coalition(learning_set.criteria_count);
      for (unsigned criterion_index = 0; criterion_index != learning_set.criteria_count; ++criterion_index) {
        const unsigned performance_rank = learning_set.performance_ranks[criterion_index][good_alternative_index];
        const auto profile_rank = profile_ranks[boundary_index][criterion_index];
        const bool is_accepted = ([&]() {
          if (learning_set.single_peaked[criterion_index]) {
            const auto [low, high] = std::get<std::pair<unsigned, unsigned>>(profile_rank);
            return low <= performance_rank && performance_rank <= high;
          } else {
            const unsigned rank = std::get<unsigned>(profile_rank);
            return performance_rank >= rank;
          }
        })();
        if (is_accepted) {
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

  const Model model = learning_set.post_process(boundaries);
  assert(count_correctly_classified_alternatives(input_problem, model, input_learning_set) == learning_set.alternatives_count);
  return model;
}

template class SatSeparationUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
