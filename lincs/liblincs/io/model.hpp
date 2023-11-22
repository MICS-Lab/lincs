// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__MODEL_HPP
#define LINCS__IO__MODEL_HPP

#include <boost/dynamic_bitset.hpp>

#include "problem.hpp"


namespace lincs {

class AcceptedValues {
 public:
  typedef std::variant<std::vector<float>, std::vector<int>, std::vector<std::string>> Thresholds;

 public:
  static AcceptedValues make_real_thresholds(const std::vector<float>& thresholds) {
    return AcceptedValues(thresholds);
  }

  static AcceptedValues make_integer_thresholds(const std::vector<int>& thresholds) {
    return AcceptedValues(thresholds);
  }

  static AcceptedValues make_enumerated_thresholds(const std::vector<std::string>& thresholds) {
    return AcceptedValues(thresholds);
  }

  // Copyable and movable
  AcceptedValues(const AcceptedValues&) = default;
  AcceptedValues& operator=(const AcceptedValues&) = default;
  AcceptedValues(AcceptedValues&&) = default;
  AcceptedValues& operator=(AcceptedValues&&) = default;

 private:
  AcceptedValues(const Thresholds& thresholds_) : thresholds(thresholds_) {}

 public:
  bool operator==(const AcceptedValues& other) const {
    return thresholds == other.thresholds;
  }

 public:
  Thresholds get_thresholds() const { return thresholds; }

  std::vector<float> get_real_thresholds() const {
    return std::get<std::vector<float>>(thresholds);
  }

  std::vector<int> get_integer_thresholds() const {
    return std::get<std::vector<int>>(thresholds);
  }

  std::vector<std::string> get_enumerated_thresholds() const {
    return std::get<std::vector<std::string>>(thresholds);
  }

 private:
  Thresholds  thresholds;
};

class SufficientCoalitions {
  // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
  // This upset can be defined:
  //   - by the weights of the criteria
  //   - explicitly by its roots

 public:
  enum class Kind { weights, roots };

 public:
  static SufficientCoalitions make_weights(const std::vector<float>& criterion_weights) {
    return SufficientCoalitions(Kind::weights, criterion_weights.size(), criterion_weights, {});
  }

  static SufficientCoalitions make_roots_from_vectors(const unsigned criteria_count, const std::vector<std::vector<unsigned>>& upset_roots) {
    return SufficientCoalitions(Kind::roots, criteria_count, {}, upset_roots);
  }

  static SufficientCoalitions make_roots_from_bitsets(const std::vector<boost::dynamic_bitset<>>& upset_roots) {
    return SufficientCoalitions(upset_roots);
  }

  // Copyable and movable
  SufficientCoalitions(const SufficientCoalitions&) = default;
  SufficientCoalitions& operator=(const SufficientCoalitions&) = default;
  SufficientCoalitions(SufficientCoalitions&&) = default;
  SufficientCoalitions& operator=(SufficientCoalitions&&) = default;

 private:
  SufficientCoalitions(
    Kind kind_,
    const unsigned criteria_count,
    const std::vector<float>& criterion_weights_,
    const std::vector<std::vector<unsigned>>& upset_roots_
  ) :
    kind(kind_),
    criterion_weights(criterion_weights_),
    upset_roots()
  {
    upset_roots.reserve(upset_roots_.size());
    for (const auto& root: upset_roots_) {
      boost::dynamic_bitset<>& upset_root = upset_roots.emplace_back(criteria_count);
      for (unsigned criterion_index: root) {
        upset_root[criterion_index] = true;
      }
    }
  }

  SufficientCoalitions(const std::vector<boost::dynamic_bitset<>>& upset_roots_) :
    kind(Kind::roots),
    criterion_weights(),
    upset_roots(upset_roots_)
  {
  }

 public:
  bool operator==(const SufficientCoalitions& other) const {
    if (kind != other.kind) {
      return false;
    }
    switch (kind) {
      case Kind::weights:
        return criterion_weights == other.criterion_weights;
      case Kind::roots:
        return upset_roots == other.upset_roots;
    }
    unreachable();
  }

 public:
  Kind get_kind() const { return kind; }

  bool is_weights() const { return kind == Kind::weights; }

  bool is_roots() const { return kind == Kind::roots; }

  std::vector<float> get_criterion_weights() const {
    assert(kind == Kind::weights);
    return criterion_weights;
  }

  std::vector<std::vector<unsigned>> get_upset_roots_as_vectors() const;

  std::vector<boost::dynamic_bitset<>> get_upset_roots_as_bitsets() const {
    assert(kind == Kind::roots);
    return upset_roots;
  }

 private:
  Kind kind;
  std::vector<float> criterion_weights;  // Indexed by [criterion_index]
  std::vector<boost::dynamic_bitset<>> upset_roots;  // Indexed by [root_coalition_index][criterion_index] and true if the criterion is in the coalition
};

// @todo(Project management, soon) Use classes with accessors everywhere. It's possible to bypass constructors of structs with uniform initialization.
struct Model {
  Model(const Problem& problem, const std::vector<AcceptedValues>& accepted_values_, const std::vector<SufficientCoalitions>& sufficient_coalitions_) :
    accepted_values(accepted_values_),
    sufficient_coalitions(sufficient_coalitions_)
  {
    assert(accepted_values.size() == problem.criteria.size());
    for (unsigned criterion_index = 0; criterion_index != problem.criteria.size(); ++criterion_index) {
      switch (problem.criteria[criterion_index].get_value_type()) {
        case Criterion::ValueType::real:
          assert(accepted_values[criterion_index].get_real_thresholds().size() == problem.ordered_categories.size() - 1);
          break;
        case Criterion::ValueType::integer:
          assert(accepted_values[criterion_index].get_integer_thresholds().size() == problem.ordered_categories.size() - 1);
          break;
        case Criterion::ValueType::enumerated:
          assert(accepted_values[criterion_index].get_enumerated_thresholds().size() == problem.ordered_categories.size() - 1);
          break;
      }
    };
    assert(sufficient_coalitions.size() == problem.ordered_categories.size() - 1);
    for (const auto& suff_coals : sufficient_coalitions) {
      switch (suff_coals.get_kind()) {
        case SufficientCoalitions::Kind::weights:
          assert(suff_coals.get_criterion_weights().size() == problem.criteria.size());
          break;
        case SufficientCoalitions::Kind::roots:
          break;
      }
    }

    // @todo(Feature, later) Check the constraints of NCS models (inclusions of sufficient coalitions, of accepted values, etc.)
    // The issue is: we're dealing with floating point data, so we need to analyse if precision loss could lead us to reject an actually correct model.
  }

  // Copyable and movable
  Model(const Model&) = default;
  Model& operator=(const Model& other) {
    accepted_values = other.accepted_values;
    sufficient_coalitions = other.sufficient_coalitions;
    return *this;
  };
  Model(Model&&) = default;
  Model& operator=(Model&& other) {
    accepted_values = std::move(other.accepted_values);
    sufficient_coalitions = std::move(other.sufficient_coalitions);
    return *this;
  }

  bool operator==(const Model& other) const {
    return accepted_values == other.accepted_values && sufficient_coalitions == other.sufficient_coalitions;
  }

  static const std::string json_schema;
  void dump(const Problem&, std::ostream&) const;
  static Model load(const Problem&, std::istream&);

  std::vector<AcceptedValues> accepted_values;
  std::vector<SufficientCoalitions> sufficient_coalitions;
};

}  // namespace lincs

#endif  // LINCS__IO__MODEL_HPP
