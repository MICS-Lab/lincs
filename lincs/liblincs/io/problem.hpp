// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <string>
#include <vector>


namespace lincs {

struct Criterion {
  std::string name;

  enum class ValueType {
    real,
    // @todo(Feature, later) Add integer
    // @todo(Feature, later) Add enumerated
  } value_type;

  enum class CategoryCorrelation {
    growing,
    decreasing,
    // @todo(Feature, later) Add single-peaked
    // @todo(Feature, later) Add single-valleyed
    // @todo(Feature, much later) Add unknown
  } category_correlation;

  float min_value;
  float max_value;

  // @todo(Project management, later) Remove these constructors
  // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
  // (Do it for other structs as well)
  Criterion() {}
  Criterion(
    const std::string& name_,
    ValueType value_type_,
    CategoryCorrelation category_correlation_,
    float min_value_,
    float max_value_
  ):
    name(name_),
    value_type(value_type_),
    category_correlation(category_correlation_),
    min_value(min_value_),
    max_value(max_value_)
  {}

  bool better_or_equal(float lhs, float rhs) const {
    switch (category_correlation) {
      case CategoryCorrelation::growing:
        return lhs >= rhs;
      case CategoryCorrelation::decreasing:
        return lhs <= rhs;
    }
    __builtin_unreachable();
  }

  bool strictly_better(float lhs, float rhs) const {
    switch (category_correlation) {
      case CategoryCorrelation::growing:
        return lhs > rhs;
      case CategoryCorrelation::decreasing:
        return lhs < rhs;
    }
    __builtin_unreachable();
  }

  // @todo(Project management, later) Remove this operator
  // The struct is usable without it in C++, and it was added only to allow using bp::vector_indexing_suite in the Python module
  // (Do it for other structs as well)
  bool operator==(const Criterion& other) const {
    return name == other.name
      && value_type == other.value_type
      && category_correlation == other.category_correlation
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
  std::vector<Category> categories;

  Problem(const std::vector<Criterion>& criteria_, const std::vector<Category>& categories_): criteria(criteria_), categories(categories_) {}

  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Problem load(std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__PROBLEM_HPP
