// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__MODEL_HPP
#define LINCS__IO__MODEL_HPP

#include <boost/dynamic_bitset.hpp>

#include "problem.hpp"


namespace lincs {

struct SufficientCoalitions {
  // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
  // This upset can be defined:
  //   - by the weights of the criteria
  //   - explicitly by its roots

  // Enum for runtime type discrimination
  enum class Kind { weights, roots } kind;

  // Tags for compile time type discrimination
  struct Weights {};
  static constexpr Weights weights = {};
  struct Roots {};
  static constexpr Roots roots = {};

  // Only one of the following two fields is used, depending on the kind.
  // Rationale for not using dynamic polymorphism: enable classification of many alternatives without the cost of virtual functions.
  // (Could/should be challenged at some point.)
  std::vector<float> criterion_weights;  // Indexed by criterion_index
  std::vector<boost::dynamic_bitset<>> upset_roots;  // Each bitset is indexed by criterion_index and is true if the criterion is in the coalition

  SufficientCoalitions(Weights, const std::vector<float>& criterion_weights_) : kind(Kind::weights), criterion_weights(criterion_weights_) {}

  SufficientCoalitions(Roots, const unsigned criteria_count, const std::vector<std::vector<unsigned>>& upset_roots_) : kind(Kind::roots), upset_roots() {
    upset_roots.reserve(upset_roots_.size());
    for (const auto& root: upset_roots_) {
      upset_roots.emplace_back(criteria_count);
      for (unsigned criterion_index: root) {
        upset_roots.back()[criterion_index] = true;
      }
    }
  }

  std::vector<std::vector<unsigned>> get_upset_roots() const {
    std::vector<std::vector<unsigned>> result;
    result.reserve(upset_roots.size());
    for (const auto& root: upset_roots) {
      result.emplace_back();
      for (unsigned criterion_index = 0; criterion_index != root.size(); ++criterion_index) {
        if (root[criterion_index]) {
          result.back().push_back(criterion_index);
        }
      }
    }
    return result;
  }
};

struct Model {
  const Problem& problem;

  struct Boundary {
    std::vector<float> profile;
    SufficientCoalitions sufficient_coalitions;

    Boundary(const std::vector<float>& profile_, const SufficientCoalitions& sufficient_coalitions_): profile(profile_), sufficient_coalitions(sufficient_coalitions_) {}

    bool operator==(const Boundary& other) const { return profile == other.profile && sufficient_coalitions.kind == other.sufficient_coalitions.kind && sufficient_coalitions.criterion_weights == other.sufficient_coalitions.criterion_weights; }
  };

  std::vector<Boundary> boundaries;  // boundary_index 0 is between category_index 0 and category_index 1

  Model(const Problem& problem_, const std::vector<Boundary>& boundaries_) : problem(problem_), boundaries(boundaries_) {}

  static const std::string json_schema;
  void dump(const Problem&, std::ostream&) const;
  static Model load(const Problem&, std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__MODEL_HPP
