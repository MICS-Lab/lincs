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
    RealThresholds(const std::vector<float>& thresholds_) : thresholds(thresholds_) {}

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
    IntegerThresholds(const std::vector<int>& thresholds_) : thresholds(thresholds_) {}

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
    EnumeratedThresholds(const std::vector<std::string>& thresholds_) : thresholds(thresholds_) {}

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

 public:
  AcceptedValues(const Self& self_) : self(self_) {}

  // Copyable and movable
  AcceptedValues(const AcceptedValues&) = default;
  AcceptedValues& operator=(const AcceptedValues&) = default;
  AcceptedValues(AcceptedValues&&) = default;
  AcceptedValues& operator=(AcceptedValues&&) = default;

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
      for (auto w : criterion_weights) {
        validate(w >= 0, "Criterion weights must be non-negative");
      }
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
    Roots(const std::vector<boost::dynamic_bitset<>>& upset_roots_) : upset_roots(upset_roots_) {}

    Roots(const unsigned criteria_count, const std::vector<std::vector<unsigned>>& upset_roots_) {
      upset_roots.reserve(upset_roots_.size());
      for (const auto& root: upset_roots_) {
        boost::dynamic_bitset<>& upset_root = upset_roots.emplace_back(criteria_count);
        for (unsigned criterion_index: root) {
          validate(criterion_index < criteria_count, "An element index in a root in a sufficient coalitions descriptor must be less than the number of criteria in the problem");
          upset_root[criterion_index] = true;
        }
      }
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
  Model(const Problem&, const std::vector<AcceptedValues>&, const std::vector<SufficientCoalitions>&);

  // Copyable and movable
  Model(const Model&) = default;
  Model& operator=(const Model&) = default;
  Model(Model&&) = default;
  Model& operator=(Model&&) = default;

 public:
  bool operator==(const Model& other) const {
    return accepted_values == other.accepted_values && sufficient_coalitions == other.sufficient_coalitions;
  }

 public:
  const std::vector<AcceptedValues>& get_accepted_values() const { return accepted_values; }
  const std::vector<SufficientCoalitions>& get_sufficient_coalitions() const { return sufficient_coalitions; }

 public:
  static const std::string json_schema;
  void dump(const Problem&, std::ostream&) const;
  static Model load(const Problem&, std::istream&);

 private:
  std::vector<AcceptedValues> accepted_values;
  std::vector<SufficientCoalitions> sufficient_coalitions;
};

}  // namespace lincs

#endif  // LINCS__IO__MODEL_HPP
