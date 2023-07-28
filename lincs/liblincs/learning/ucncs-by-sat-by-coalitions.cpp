// Copyright 2023 Vincent Jacques

#include "ucncs-by-sat-by-coalitions.hpp"

#include <algorithm>
#include <map>
#include <type_traits>

#include "exception.hpp"
#include "../sat/minisat.hpp"


namespace lincs {

template<typename SatProblem>
Model SatCoalitionsUcncsLearning<SatProblem>::perform() {
  const unsigned criteria_count = problem.criteria.size();
  const unsigned categories_count = problem.categories.size();
  const unsigned boundaries_count = categories_count - 1;

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

  SatProblem sat;

  // x[i][h][k]: value k is above profile h on criterion i
  std::vector<std::vector<std::vector<typename SatProblem::variable_type>>> x(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    x[criterion_index].resize(categories_count);
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      x[criterion_index][boundary_index].resize(unique_values[criterion_index].size());
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        x[criterion_index][boundary_index][value_index] = sat.create_variable();
      }
    }
  }

  // A subsets b of criteria (i.e. a coalition) is represented
  // as a bitset where bit i is set if and only if criteria i is in b.
  // y[b]: coalition b is sufficient
  unsigned subsets_count = 1 << criteria_count;
  std::vector<typename SatProblem::variable_type> y(subsets_count);
  for (unsigned subset = 0; subset != subsets_count; ++subset) {
    y[subset] = sat.create_variable();
  }

  sat.mark_all_variables_created();

  // Note: "A => B" <=> "-A or B"

  // Ascending scales: values are ordered according to index k
  // so if a value is above profile h, then values above it are above profile h too
  // so x[i][h][k] => x[i][h][k + 1]
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size() - 1; ++value_index) {
        sat.add_clause({-x[criterion_index][boundary_index][value_index], x[criterion_index][boundary_index][value_index + 1]});
      }
    }
  }

  // Hierarchy of profiles: profiles are ordered according to index h
  // so if a value is above a profile h, then it is above lower profiles too
  // so x[i][h][k] => x[i][h - 1][k]
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
      for (unsigned boundary_index_a = 1; boundary_index_a != boundaries_count; ++boundary_index_a) {
        sat.add_clause({-x[criterion_index][boundary_index_a][value_index], x[criterion_index][boundary_index_a - 1][value_index]});
      }
    }
  }

  // Coalitions strength: coalitions are an upset
  // so if coalition b is sufficient and b is included in bp, then bp is sufficient too
  // so y[b] => y[bp] for b included in bp
  // @todo Optimize this nested loop on pairs of coalitions using the fact that b is included in bp
  for (unsigned subset_a = 0; subset_a != subsets_count; ++subset_a) {
    for (unsigned subset_b = 0; subset_b != subsets_count; ++subset_b) {
      // "subset_a included in subset_b" <=> "all bits set in subset_a are set in subset_b"
      if ((subset_a & subset_b) == subset_a && subset_a != subset_b) {
        sat.add_clause({-y[subset_a], y[subset_b]});
      }
    }
  }

  // Alternatives are outranked by boundary above them
  for (auto alternative : learning_set.alternatives) {
    const unsigned category_index = *alternative.category_index;
    if (category_index == problem.categories.size() - 1) {
      continue;
    }
    for (unsigned subset = 0; subset != subsets_count; ++subset) {
      std::vector<typename SatProblem::variable_type> clause;
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        // "criterion_index in subset" <=> "bit criterion_index is set in subset"
        if (subset & (1 << criterion_index)) {
          const auto lb = std::lower_bound(unique_values[criterion_index].begin(), unique_values[criterion_index].end(), alternative.profile[criterion_index]);
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = lb - unique_values[criterion_index].begin();
          assert(criterion_index < x.size());
          assert(category_index < x[criterion_index].size());
          assert(value_index < x[criterion_index][category_index].size());
          clause.push_back(-x[criterion_index][category_index][value_index]);
        }
      }
      clause.push_back(-y[subset]);
      sat.add_clause(clause);
    }
  }

  // Alternatives outrank the boundary below them
  for (auto alternative : learning_set.alternatives) {
    const unsigned category_index = *alternative.category_index;
    if (category_index == 0) {
      continue;
    }
    for (unsigned subset = 0; subset != subsets_count; ++subset) {
      std::vector<typename SatProblem::variable_type> clause;
      for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        // "criterion_index in subset" <=> "bit criterion_index is set in subset"
        if (subset & (1 << criterion_index)) {
          const auto lb = std::lower_bound(unique_values[criterion_index].begin(), unique_values[criterion_index].end(), alternative.profile[criterion_index]);
          assert(lb != unique_values[criterion_index].end());
          const unsigned value_index = lb - unique_values[criterion_index].begin();
          assert(criterion_index < x.size());
          assert(category_index - 1 < x[criterion_index].size());
          assert(value_index < x[criterion_index][category_index - 1].size());
          clause.push_back(x[criterion_index][category_index - 1][value_index]);
        }
      }
      unsigned subset_complement = ~subset & (subsets_count - 1);
      clause.push_back(y[subset_complement]);
      sat.add_clause(clause);
    }
  }

  std::optional<std::vector<bool>> solution = sat.solve();

  if (!solution) {
    throw LearningFailureException();
  }

  std::vector<std::vector<unsigned>> roots;
  for (unsigned subset_a = 0; subset_a != subsets_count; ++subset_a) {
    if ((*solution)[y[subset_a]]) {
      bool is_root = true;
      // @todo Optimize this search for actual roots; it may be something like a transitive reduction
      for (unsigned subset_b = 0; subset_b != subsets_count; ++subset_b) {
        if ((*solution)[y[subset_b]]) {
          if ((subset_a & subset_b) == subset_b && subset_a != subset_b) {
            is_root = false;
            break;
          }
        }
      }
      if (is_root) {
        roots.emplace_back();
        for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
          if (subset_a & (1 << criterion_index)) {
            roots.back().push_back(criterion_index);
          }
        }
      }
    }
  }

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(boundaries_count);
  for (unsigned boundary_index = 0; boundary_index != boundaries_count; ++boundary_index) {
    std::vector<float> profile(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      bool found = false;
      // @todo Replace next loop with a binary search
      for (unsigned value_index = 0; value_index != unique_values[criterion_index].size(); ++value_index) {
        if ((*solution)[x[criterion_index][boundary_index][value_index]]) {
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

    boundaries.emplace_back(profile, SufficientCoalitions{SufficientCoalitions::roots, criteria_count, roots});
  }

  return Model{problem, boundaries};
}

template class SatCoalitionsUcncsLearning<MinisatSatProblem>;

}  // namespace lincs
