// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__IO__MODEL_HPP
#define LINCS__IO__MODEL_HPP

#include <optional>
#include <utility>

#include <boost/dynamic_bitset.hpp>

#include "../internal.hpp"
#include "problem.hpp"


namespace lincs {

class AcceptedValues {
 public:
  class RealThresholds {
   public:
    RealThresholds(const std::vector<std::optional<float>>& thresholds_) : thresholds(thresholds_) {}

   public:
    bool operator==(const RealThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<std::optional<float>>& get_thresholds() const { return thresholds; }

   private:
    std::vector<std::optional<float>> thresholds;
  };

  class IntegerThresholds {
   public:
    IntegerThresholds(const std::vector<std::optional<int>>& thresholds_) : thresholds(thresholds_) {}

   public:
    bool operator==(const IntegerThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<std::optional<int>>& get_thresholds() const { return thresholds; }

   private:
    std::vector<std::optional<int>> thresholds;
  };

  class EnumeratedThresholds {
   public:
    EnumeratedThresholds(const std::vector<std::optional<std::string>>& thresholds_) : thresholds(thresholds_) {}

   public:
    bool operator==(const EnumeratedThresholds& other) const {
      return thresholds == other.thresholds;
    }

   public:
    const std::vector<std::optional<std::string>>& get_thresholds() const { return thresholds; }

   private:
    std::vector<std::optional<std::string>> thresholds;
  };

  class RealIntervals {
   public:
    RealIntervals(const std::vector<std::optional<std::pair<float, float>>>& intervals_) : intervals(intervals_) {}

   public:
    bool operator==(const RealIntervals& other) const {
      return intervals == other.intervals;
    }

   public:
    const std::vector<std::optional<std::pair<float, float>>>& get_intervals() const { return intervals; }

   private:
    std::vector<std::optional<std::pair<float, float>>> intervals;
  };

  class IntegerIntervals {
   public:
    IntegerIntervals(const std::vector<std::optional<std::pair<int, int>>>& intervals_) : intervals(intervals_) {}

   public:
    bool operator==(const IntegerIntervals& other) const {
      return intervals == other.intervals;
    }

   public:
    const std::vector<std::optional<std::pair<int, int>>>& get_intervals() const { return intervals; }

   private:
    std::vector<std::optional<std::pair<int, int>>> intervals;
  };

  // WARNING: keep the enum and the variant consistent with 'Criterion::ValueType', 'get_kind', and 'get_value_type'
  enum class Kind { thresholds, intervals };
  typedef std::variant<
    RealThresholds,
    IntegerThresholds,
    EnumeratedThresholds,
    RealIntervals,
    IntegerIntervals
  > Self;

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
  Kind get_kind() const {
    return dispatch(
      self,
      [](const RealThresholds&) { return Kind::thresholds; },
      [](const IntegerThresholds&) { return Kind::thresholds; },
      [](const EnumeratedThresholds&) { return Kind::thresholds; },
      [](const RealIntervals&) { return Kind::intervals; },
      [](const IntegerIntervals&) { return Kind::intervals; }
    );
  }
  Criterion::ValueType get_value_type() const {
    return dispatch(
      self,
      [](const RealThresholds&) { return Criterion::ValueType::real; },
      [](const IntegerThresholds&) { return Criterion::ValueType::integer; },
      [](const EnumeratedThresholds&) { return Criterion::ValueType::enumerated; },
      [](const RealIntervals&) { return Criterion::ValueType::real; },
      [](const IntegerIntervals&) { return Criterion::ValueType::integer; }
    );
  }
  const Self& get() const { return self; }

  bool is_real() const { return get_value_type() == Criterion::ValueType::real; }
  bool is_integer() const { return get_value_type() == Criterion::ValueType::integer; }
  bool is_enumerated() const { return get_value_type() == Criterion::ValueType::enumerated; }

  bool is_thresholds() const { return get_kind() == Kind::thresholds; }
  bool is_intervals() const { return get_kind() == Kind::intervals; }

  const RealThresholds& get_real_thresholds() const {
    return std::get<RealThresholds>(self);
  }

  const IntegerThresholds& get_integer_thresholds() const {
    return std::get<IntegerThresholds>(self);
  }

  const EnumeratedThresholds& get_enumerated_thresholds() const {
    return std::get<EnumeratedThresholds>(self);
  }

  const RealIntervals& get_real_intervals() const {
    return std::get<RealIntervals>(self);
  }

  const IntegerIntervals& get_integer_intervals() const {
    return std::get<IntegerIntervals>(self);
  }

 private:
  Self self;
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

    bool accept(const boost::dynamic_bitset<>& coalition) const {
      float sum = 0;
      for (unsigned criterion_index = 0; criterion_index < criterion_weights.size(); ++criterion_index) {
        if (coalition[criterion_index]) {
          sum += criterion_weights[criterion_index];
        }
      }
      return sum >= 1;
    }

   private:
    std::vector<float> criterion_weights;  // Indexed by [criterion_index]
  };

  class Roots {
   public:
    Roots(const Problem& problem, const std::vector<std::vector<unsigned>>& upset_roots_) :
      Roots(Internal(), problem.get_criteria().size(), upset_roots_) {}

    Roots(Internal, const std::vector<boost::dynamic_bitset<>>& upset_roots_) :
      upset_roots(upset_roots_) {}

    Roots(Internal, const unsigned criteria_count, const std::vector<std::vector<unsigned>>& upset_roots_) {
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

    bool accept(const boost::dynamic_bitset<>& coalition) const {
      for (const auto& root: upset_roots) {
        if ((coalition & root) == root) {
          return true;
        }
      }
      return false;
    }

   private:
    std::vector<boost::dynamic_bitset<>> upset_roots;  // Indexed by [root_coalition_index][criterion_index] and true if the criterion is in the coalition
  };

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
  Kind get_kind() const {
    return dispatch(
      self,
      [](const Weights&) { return Kind::weights; },
      [](const Roots&) { return Kind::roots; }
    );
  }
  const Self& get() const { return self; }

  bool is_weights() const { return get_kind() == Kind::weights; }
  const Weights& get_weights() const { return std::get<Weights>(self); }

  bool is_roots() const { return get_kind() == Kind::roots; }
  const Roots& get_roots() const { return std::get<Roots>(self); }

 private:
  Self self;
};

class Model {
 public:
  Model(const Problem&, const std::vector<AcceptedValues>&, const std::vector<SufficientCoalitions>&);

  Model(Internal, const std::vector<AcceptedValues>& accepted_values_, const std::vector<SufficientCoalitions>& sufficient_coalitions_) :
    accepted_values(accepted_values_),
    sufficient_coalitions(sufficient_coalitions_)
  {}

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
