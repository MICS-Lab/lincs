// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__MODEL_HPP
#define LINCS__IO__MODEL_HPP

#include "problem.hpp"


namespace lincs {

struct Model {
  const Problem& problem;

  struct SufficientCoalitions {
    // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
    // This upset can be defined:
    enum class Kind {
      weights,  // by the weights of the criteria
      // @todo Add upset_roots,  // explicitly by its roots
    } kind;

    std::vector<float> criterion_weights;

    SufficientCoalitions() {};
    SufficientCoalitions(Kind kind_, const std::vector<float>& criterion_weights_): kind(kind_), criterion_weights(criterion_weights_) {}
  };

  struct Boundary {
    std::vector<float> profile;
    SufficientCoalitions sufficient_coalitions;

    Boundary() {};
    Boundary(const std::vector<float>& profile_, const SufficientCoalitions& sufficient_coalitions_): profile(profile_), sufficient_coalitions(sufficient_coalitions_) {}

    bool operator==(const Boundary& other) const { return profile == other.profile && sufficient_coalitions.kind == other.sufficient_coalitions.kind && sufficient_coalitions.criterion_weights == other.sufficient_coalitions.criterion_weights; }
  };

  std::vector<Boundary> boundaries;  // boundary_index 0 is between category_index 0 and category_index 1

  Model(const Problem& problem_, const std::vector<Boundary>& boundaries_) : problem(problem_), boundaries(boundaries_) {}

  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Model load(const Problem&, std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__MODEL_HPP
