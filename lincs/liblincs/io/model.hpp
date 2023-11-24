// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__MODEL_HPP
#define LINCS__IO__MODEL_HPP

#include <boost/dynamic_bitset.hpp>

#include "problem.hpp"


namespace lincs {

class AcceptedValues {
 public:
  class RealThresholds {
   public:
    RealThresholds(const std::vector<float>& thresholds_) : thresholds(thresholds_) {
      // @todo(Feature, v1.1) Validate thresholds (e.g. they are ordered)
    }

   public:
    bool operator==(const RealThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<float>& get_thresholds() const { return thresholds; }

   private:
    std::vector<float> thresholds;
  };

  class IntegerThresholds {
   public:
    IntegerThresholds(const std::vector<int>& thresholds_) : thresholds(thresholds_) {
      // @todo(Feature, v1.1) Validate thresholds (e.g. they are ordered)
    }

   public:
    bool operator==(const IntegerThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<int>& get_thresholds() const { return thresholds; }

   private:
    std::vector<int> thresholds;
  };

  class EnumeratedThresholds {
   public:
    EnumeratedThresholds(const std::vector<std::string>& thresholds_) : thresholds(thresholds_) {
      // @todo(Feature, v1.1) Validate thresholds (e.g. they are ordered)
    }

   public:
    bool operator==(const EnumeratedThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<std::string>& get_thresholds() const { return thresholds; }

   private:
    std::vector<std::string> thresholds;
  };

  // WARNING: keep the enum and the variant consistent with 'Criterion::ValueType'
  // WARNING: Adding a value to the enum will require fixing 'get_value_type' (and obviously 'get_kind' and 'is_thresholds')
  enum class Kind { thresholds };
  typedef std::variant<RealThresholds, IntegerThresholds, EnumeratedThresholds> Self;

  // Copyable and movable
  AcceptedValues(const AcceptedValues&) = default;
  AcceptedValues& operator=(const AcceptedValues&) = default;
  AcceptedValues(AcceptedValues&&) = default;
  AcceptedValues& operator=(AcceptedValues&&) = default;

 public:
  AcceptedValues(const Self& self_) : self(self_) {}

 public:
  bool operator==(const AcceptedValues& other) const {
    return self == other.self;
  }

 public:
  Kind get_kind() const { return Kind::thresholds; }
  Criterion::ValueType get_value_type() const { return Criterion::ValueType(self.index()); }
  const Self& get() const { return self; }

  bool is_real() const { return get_value_type() == Criterion::ValueType::real; }
  bool is_thresholds() const { return true; }
  const RealThresholds& get_real_thresholds() const {
    return std::get<RealThresholds>(self);
  }

  bool is_integer() const { return get_value_type() == Criterion::ValueType::integer; }
  const IntegerThresholds& get_integer_thresholds() const {
    return std::get<IntegerThresholds>(self);
  }

  bool is_enumerated() const { return get_value_type() == Criterion::ValueType::enumerated; }
  const EnumeratedThresholds& get_enumerated_thresholds() const {
    return std::get<EnumeratedThresholds>(self);
  }

 private:
  Self self;  // @todo(Feature, v1.1) Evaluate wether we could remove the class and use directly the variant. Are there any potential future attributes to be added?
};

class SufficientCoalitions {
  // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
  // This upset can be defined:
  //   - by the weights of the criteria
  //   - explicitly by its roots

 public:
  class Weights {
   public:
    Weights(const std::vector<float>& criterion_weights_) : criterion_weights(criterion_weights_) {
      // @todo(Feature, v1.1) Validate criterion_weights?
    }

   public:
    bool operator==(const Weights& other) const {
      return criterion_weights == other.criterion_weights;
    }

   public:
    const std::vector<float>& get_criterion_weights() const { return criterion_weights; }

   private:
    std::vector<float> criterion_weights;  // Indexed by [criterion_index]
  };

  class Roots {
   public:
    Roots(const std::vector<boost::dynamic_bitset<>>& upset_roots_) : upset_roots(upset_roots_) {
      // @todo(Feature, v1.1) Validate upset_roots?
    }

    Roots(const unsigned criteria_count, const std::vector<std::vector<unsigned>>& upset_roots_) {
      upset_roots.reserve(upset_roots_.size());
      for (const auto& root: upset_roots_) {
        boost::dynamic_bitset<>& upset_root = upset_roots.emplace_back(criteria_count);
        for (unsigned criterion_index: root) {
          upset_root[criterion_index] = true;
        }
      }
      // @todo(Feature, v1.1) Validate upset_roots?
    }

   public:
    bool operator==(const Roots& other) const {
      return upset_roots == other.upset_roots;
    }

   public:
    std::vector<std::vector<unsigned>> get_upset_roots_as_vectors() const;

    std::vector<boost::dynamic_bitset<>> get_upset_roots_as_bitsets() const {
      return upset_roots;
    }

   private:
    std::vector<boost::dynamic_bitset<>> upset_roots;  // Indexed by [root_coalition_index][criterion_index] and true if the criterion is in the coalition
  };

  // WARNING: keep the enum and the variant consistent
  // (because the variant's index is used as the enum's value)
  enum class Kind { weights, roots };
  typedef std::variant<Weights, Roots> Self;

 public:
  SufficientCoalitions(const Self& self_) : self(self_) {}

 public:
  // Copyable and movable
  SufficientCoalitions(const SufficientCoalitions&) = default;
  SufficientCoalitions& operator=(const SufficientCoalitions&) = default;
  SufficientCoalitions(SufficientCoalitions&&) = default;
  SufficientCoalitions& operator=(SufficientCoalitions&&) = default;

 public:
  bool operator==(const SufficientCoalitions& other) const {
    return self == other.self;
  }

 public:
  Kind get_kind() const { return Kind(self.index()); }
  const Self& get() const { return self; }

  bool is_weights() const { return get_kind() == Kind::weights; }
  const Weights& get_weights() const { return std::get<Weights>(self); }

  bool is_roots() const { return get_kind() == Kind::roots; }
  const Roots& get_roots() const { return std::get<Roots>(self); }

 private:
  Self self;  // @todo(Feature, v1.1) Evaluate wether we could remove the class and use directly the variant. Are there any potential future attributes to be added?
};

class Model {
 public:
  Model(const Problem& problem, const std::vector<AcceptedValues>& accepted_values_, const std::vector<SufficientCoalitions>& sufficient_coalitions_) :
    accepted_values(accepted_values_),
    sufficient_coalitions(sufficient_coalitions_)
  {
    // @todo(Feature, v1.1) Use 'lincs::validate' instead of 'assert': this validation must occur in release mode as well
    assert(accepted_values.size() == problem.criteria.size());
    for (unsigned criterion_index = 0; criterion_index != problem.criteria.size(); ++criterion_index) {
      dispatch(
        accepted_values[criterion_index].get(),
        [&problem](const AcceptedValues::RealThresholds& thresholds) {
          assert(thresholds.get_thresholds().size() == problem.ordered_categories.size() - 1);
        },
        [&problem](const AcceptedValues::IntegerThresholds& thresholds) {
          assert(thresholds.get_thresholds().size() == problem.ordered_categories.size() - 1);
        },
        [&problem](const AcceptedValues::EnumeratedThresholds& thresholds) {
          assert(thresholds.get_thresholds().size() == problem.ordered_categories.size() - 1);
        }
      );
    };
    assert(sufficient_coalitions.size() == problem.ordered_categories.size() - 1);
    for (const auto& sufficient_coalitions_ : sufficient_coalitions) {
      dispatch(
        sufficient_coalitions_.get(),
        [&](const SufficientCoalitions::Weights& weights) {
          assert(weights.get_criterion_weights().size() == problem.criteria.size());
        },
        [&](const SufficientCoalitions::Roots& roots) {
          // Nothing to do
        }
      );
    }

    // @todo(Feature, v1.1) Check the constraints of NCS models (inclusions of sufficient coalitions, of accepted values, etc.)
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

 public:
  bool operator==(const Model& other) const {
    return accepted_values == other.accepted_values && sufficient_coalitions == other.sufficient_coalitions;
  }

 public:
  static const std::string json_schema;
  void dump(const Problem&, std::ostream&) const;
  static Model load(const Problem&, std::istream&);

 public:
  std::vector<AcceptedValues> accepted_values;
  std::vector<SufficientCoalitions> sufficient_coalitions;
};

}  // namespace lincs

#endif  // LINCS__IO__MODEL_HPP
