// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <string>
#include <vector>

#include "../unreachable.hpp"


namespace lincs {

struct Criterion {
  std::string name;

  enum class ValueType {
    real,
    // @todo(Feature, later) Add integer, with min and max
    // @todo(Feature, later) Add enumerated, with ordered_values
  } value_type;

  enum class PreferenceDirection {
    increasing,
    isotone=increasing,
    decreasing,
    antitone=decreasing,
    // @todo(Feature, later) Add single-peaked
    // @todo(Feature, much later) Add unknown
  } preference_direction;

  float min_value;
  float max_value;

  // @todo(Project management, later) Remove these constructors
  // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
  // (Do it for other structs as well)
  Criterion() {}
  Criterion(
    const std::string& name_,
    ValueType value_type_,
    PreferenceDirection preference_direction_,
    float min_value_,
    float max_value_
  ):
    name(name_),
    value_type(value_type_),
    preference_direction(preference_direction_),
    min_value(min_value_),
    max_value(max_value_)
  {}

  // @todo(Project management, later) Remove this operator
  // The struct is usable without it in C++, and it was added only to allow using bp::vector_indexing_suite in the Python module
  // (Do it for other structs as well)
  bool operator==(const Criterion& other) const {
    return name == other.name
      && value_type == other.value_type
      && preference_direction == other.preference_direction
      && min_value == other.min_value
      && max_value == other.max_value;
  }
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
