// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <any>
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
  static Criterion make_real(const std::string& name, PreferenceDirection preference_direction, float min_value, float max_value) {
    return Criterion(name, ValueType::real, preference_direction, min_value, max_value, {});
  }

  static Criterion make_integer(const std::string& name, PreferenceDirection preference_direction, int min_value, int max_value) {
    return Criterion(name, ValueType::integer, preference_direction, min_value, max_value, {});
  }

  static Criterion make_enumerated(const std::string& name, const std::vector<std::string>& ordered_values) {
    return Criterion(name, ValueType::enumerated, PreferenceDirection::increasing, {}, {}, ordered_values);
  }

 private:
  Criterion(
    const std::string& name_,
    ValueType value_type_,
    PreferenceDirection preference_direction_,
    std::any min_value_,
    std::any max_value_,
    const std::vector<std::string>& ordered_values_
  ):
    name(name_),
    value_type(value_type_),
    preference_direction(preference_direction_),
    min_value(min_value_),
    max_value(max_value_),
    ordered_values(ordered_values_)
  {}

 public:
  bool operator==(const Criterion& other) const {
    if (value_type != other.value_type) {
      return false;
    }
    switch (value_type) {
      case ValueType::real:
        return
          name == other.name &&
          preference_direction == other.preference_direction &&
          std::any_cast<float>(min_value) == std::any_cast<float>(other.min_value) &&
          std::any_cast<float>(max_value) == std::any_cast<float>(other.max_value);
      case ValueType::integer:
        return
          name == other.name &&
          preference_direction == other.preference_direction &&
          std::any_cast<int>(min_value) == std::any_cast<int>(other.min_value) &&
          std::any_cast<int>(max_value) == std::any_cast<int>(other.max_value);
      case ValueType::enumerated:
        return
          name == other.name &&
          ordered_values == other.ordered_values;
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

  void set_preference_direction__for_tests(PreferenceDirection preference_direction_) {
    assert(is_real() || is_integer());
    preference_direction = preference_direction_;
  }

  float get_real_min_value() const {
    assert(is_real());
    return std::any_cast<float>(min_value);
  }

  float get_real_max_value() const {
    assert(is_real());
    return std::any_cast<float>(max_value);
  }

  int get_integer_min_value() const {
    assert(is_integer());
    return std::any_cast<int>(min_value);
  }

  int get_integer_max_value() const {
    assert(is_integer());
    return std::any_cast<int>(max_value);
  }

  std::vector<std::string> get_ordered_values() const {
    assert(is_enumerated());
    return ordered_values;
  }

 private:
  std::string name;
  ValueType value_type;
  PreferenceDirection preference_direction;
  std::any min_value;
  std::any max_value;
  std::vector<std::string> ordered_values;
};

struct Category {
  std::string name;

  Category() {}
  Category(const std::string& name_): name(name_) {}

  bool operator==(const Category& other) const { return name == other.name; }
};

struct Problem {
  std::vector<Criterion> criteria;
  std::vector<Category> ordered_categories;

  Problem(const std::vector<Criterion>& criteria_, const std::vector<Category>& ordered_categories_): criteria(criteria_), ordered_categories(ordered_categories_) {}

  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Problem load(std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__PROBLEM_HPP
