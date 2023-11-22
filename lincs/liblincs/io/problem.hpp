// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "../variant-dispatch.hpp"
#include "../unreachable.hpp"


namespace lincs {

class Criterion {
 public:
  enum class PreferenceDirection {
    increasing,
    isotone=increasing,
    decreasing,
    antitone=decreasing,
    // @todo(Feature, later) Add single-peaked
    // @todo(Feature, much later) Add unknown
  };

  struct RealValues {
    PreferenceDirection preference_direction;
    float min_value;
    float max_value;

    bool operator==(const RealValues& other) const {
      return preference_direction == other.preference_direction && min_value == other.min_value && max_value == other.max_value;
    }
  };

  struct IntegerValues {
    PreferenceDirection preference_direction;
    int min_value;
    int max_value;

    bool operator==(const IntegerValues& other) const {
      return preference_direction == other.preference_direction && min_value == other.min_value && max_value == other.max_value;
    }
  };

  struct EnumeratedValues {
    std::vector<std::string> ordered_values;
    std::map<std::string, unsigned> value_ranks;

    EnumeratedValues(const std::vector<std::string>& ordered_values_) : ordered_values(ordered_values_) {
      for (unsigned i = 0; i < ordered_values.size(); ++i) {
        value_ranks[ordered_values[i]] = i;
      }
    }

    bool operator==(const EnumeratedValues& other) const {
      return ordered_values == other.ordered_values;  // No need to compare value_ranks
    }
  };

  // WARNING: keep the enum and the variant consistent
  // (because the variant's index is used as the enum's value)
  enum class ValueType { real, integer, enumerated };
  typedef std::variant<RealValues, IntegerValues, EnumeratedValues> Values;
  // END OF WARNING

 public:
  static Criterion make_real(const std::string& name, PreferenceDirection preference_direction, float real_min_value, float real_max_value) {
    assert(real_min_value <= real_max_value);
    return Criterion(name, RealValues{preference_direction, real_min_value, real_max_value});
  }

  static Criterion make_integer(const std::string& name, PreferenceDirection preference_direction, int int_min_value, int int_max_value) {
    assert(int_min_value <= int_max_value);
    return Criterion(name, IntegerValues{preference_direction, int_min_value, int_max_value});
  }

  static Criterion make_enumerated(const std::string& name, const std::vector<std::string>& ordered_values) {
    assert(ordered_values.size() > 0);
    return Criterion(name, EnumeratedValues{ordered_values});
  }

 private:
  Criterion(const std::string& name_, Values values_) : name(name_), values(values_) {}

 public:
  bool operator==(const Criterion& other) const {
    return name == other.name && values == other.values;
  }

  // @todo(Project management, soon) Return const ref
  std::string get_name() const { return name; }

  ValueType get_value_type() const { return ValueType(values.index()); }

  bool is_real() const { return get_value_type() == ValueType::real; }

  bool is_integer() const { return get_value_type() == ValueType::integer; }

  bool is_enumerated() const { return get_value_type() == ValueType::enumerated; }

  const Values& get_values() const { return values; }

  PreferenceDirection get_preference_direction() const {
    if (is_real()) {
      return std::get<RealValues>(values).preference_direction;
    } else {
      return std::get<IntegerValues>(values).preference_direction;
    }
  }

  bool is_increasing() const {
    return get_preference_direction() == PreferenceDirection::increasing;
  }

  bool is_decreasing() const {
    return get_preference_direction() == PreferenceDirection::decreasing;
  }

  float get_real_min_value() const {
    return std::get<RealValues>(values).min_value;
  }

  float get_real_max_value() const {
    return std::get<RealValues>(values).max_value;
  }

  int get_integer_min_value() const {
    return std::get<IntegerValues>(values).min_value;
  }

  int get_integer_max_value() const {
    return std::get<IntegerValues>(values).max_value;
  }

  std::vector<std::string> get_ordered_values() const {
    return std::get<EnumeratedValues>(values).ordered_values;
  }

  std::map<std::string, unsigned> get_value_ranks() const {
    return std::get<EnumeratedValues>(values).value_ranks;
  }

  unsigned get_value_rank(const std::string& value) const {
    return std::get<EnumeratedValues>(values).value_ranks.at(value);
  }

 private:
  std::string name;
  Values values;
};

class Category {
 public:
  Category(const std::string& name_): name(name_) {}

 public:
  bool operator==(const Category& other) const { return name == other.name; }

 public:
  std::string name;
};

class Problem {
 public:
  Problem(const std::vector<Criterion>& criteria_, const std::vector<Category>& ordered_categories_): criteria(criteria_), ordered_categories(ordered_categories_) {
    assert(criteria.size() > 0);
    assert(ordered_categories.size() >= 2);
  }

 public:
  bool operator==(const Problem& other) const {
    return criteria == other.criteria && ordered_categories == other.ordered_categories;
  }

 public:
  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Problem load(std::istream&);

 public:
  std::vector<Criterion> criteria;
  std::vector<Category> ordered_categories;
};

}  // namespace lincs

#endif  // LINCS__IO__PROBLEM_HPP
