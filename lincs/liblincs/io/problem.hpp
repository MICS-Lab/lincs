// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <cassert>
#include <string>
#include <vector>

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

  enum class ValueType {
    real,
    integer,
    enumerated
  };

 public:
  static Criterion make_real(const std::string& name, PreferenceDirection preference_direction, float real_min_value, float real_max_value) {
    assert(real_min_value <= real_max_value);
    return Criterion(name, ValueType::real, preference_direction, real_min_value, real_max_value, 0, 0, {});
  }

  static Criterion make_integer(const std::string& name, PreferenceDirection preference_direction, int int_min_value, int int_max_value) {
    assert(int_min_value <= int_max_value);
    return Criterion(name, ValueType::integer, preference_direction, 0, 0, int_min_value, int_max_value, {});
  }

  static Criterion make_enumerated(const std::string& name, const std::vector<std::string>& ordered_values) {
    assert(ordered_values.size() > 0);
    return Criterion(name, ValueType::enumerated, PreferenceDirection::increasing, 0, 0, 0, 0, ordered_values);
  }

 private:
  Criterion(
    const std::string& name_,
    ValueType value_type_,
    PreferenceDirection preference_direction_,
    float real_min_value_,
    float real_max_value_,
    int int_min_value_,
    int int_max_value_,
    const std::vector<std::string>& ordered_values_
  ):
    name(name_),
    value_type(value_type_),
    preference_direction(preference_direction_),
    real_min_value(real_min_value_),
    real_max_value(real_max_value_),
    int_min_value(int_min_value_),
    int_max_value(int_max_value_),
    ordered_values(ordered_values_)
  {}

 public:
  bool operator==(const Criterion& other) const {
    if (value_type != other.value_type || name != other.name) {
      return false;
    }
    switch (value_type) {
      case ValueType::real:
        return preference_direction == other.preference_direction &&
          real_min_value == other.real_min_value && real_max_value == other.real_max_value;
      case ValueType::integer:
        return preference_direction == other.preference_direction &&
          int_min_value == other.int_min_value && int_max_value == other.int_max_value;
      case ValueType::enumerated:
        return ordered_values == other.ordered_values;
    }
    unreachable();
  }

  std::string get_name() const { return name; }
  
  ValueType get_value_type() const { return value_type; }

  bool is_real() const { return value_type == ValueType::real; }

  bool is_integer() const { return value_type == ValueType::integer; }

  bool is_enumerated() const { return value_type == ValueType::enumerated; }

  PreferenceDirection get_preference_direction() const {
    assert(is_real() || is_integer());
    return preference_direction;
  }

  bool is_increasing() const {
    assert(is_real() || is_integer());
    return preference_direction == PreferenceDirection::increasing;
  }

  bool is_decreasing() const {
    assert(is_real() || is_integer());
    return preference_direction == PreferenceDirection::decreasing;
  }

  float get_real_min_value() const {
    assert(is_real());
    return real_min_value;
  }

  float get_real_max_value() const {
    assert(is_real());
    return real_max_value;
  }

  int get_integer_min_value() const {
    assert(is_integer());
    return int_min_value;
  }

  int get_integer_max_value() const {
    assert(is_integer());
    return int_max_value;
  }

  std::vector<std::string> get_ordered_values() const {
    assert(is_enumerated());
    return ordered_values;
  }

 private:
  std::string name;
  ValueType value_type;
  // @todo(Project management, later) Use 'union' or equivalent to store only the relevant values
  PreferenceDirection preference_direction;  // Only for real and integer
  float real_min_value;  // Only for real
  float real_max_value;
  int int_min_value;  // Only for integer
  int int_max_value;
  std::vector<std::string> ordered_values;  // Only for enumerated
};

struct Category {
  std::string name;

  Category() {}
  Category(const std::string& name_): name(name_) {}

  bool operator==(const Category& other) const { return name == other.name; }
};

struct Problem {
  Problem(const std::vector<Criterion>& criteria_, const std::vector<Category>& ordered_categories_): criteria(criteria_), ordered_categories(ordered_categories_) {
    assert(criteria.size() > 0);
    assert(ordered_categories.size() >= 2);
  }

  bool operator==(const Problem& other) const {
    return criteria == other.criteria && ordered_categories == other.ordered_categories;
  }

  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Problem load(std::istream&);

  std::vector<Criterion> criteria;
  std::vector<Category> ordered_categories;
};

}  // namespace lincs

#endif  // LINCS__IO__PROBLEM_HPP
