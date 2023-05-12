// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__DOMAIN_HPP
#define LINCS__IO__DOMAIN_HPP

#include <string>
#include <vector>


namespace lincs {

struct Domain {
  struct Criterion {
    std::string name;

    enum class ValueType {
      real,
      // @todo Add integer
      // @todo Add enumerated
    } value_type;

    enum class CategoryCorrelation {
      growing,
      // @todo Add decreasing
      // @todo Add single-peaked
      // @todo Add single-valleyed
      // @todo Add unknown
    } category_correlation;

    // @todo Remove these constructors
    // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
    // (Do it for other structs as well)
    Criterion() {}
    Criterion(const std::string& name_, ValueType value_type_, CategoryCorrelation category_correlation_): name(name_), value_type(value_type_), category_correlation(category_correlation_) {}

    // @todo Remove this operator
    // The struct is usable without it in C++, and it was added only to allow using bp::vector_indexing_suite in the Python module
    // (Do it for other structs as well)
    bool operator==(const Criterion& other) const {
      return name == other.name && value_type == other.value_type && category_correlation == other.category_correlation;
    }
  };

  std::vector<Criterion> criteria;

  struct Category {
    std::string name;

    Category() {}
    Category(const std::string& name_): name(name_) {}

    bool operator==(const Category& other) const { return name == other.name; }
  };

  std::vector<Category> categories;

  Domain(const std::vector<Criterion>& criteria_, const std::vector<Category>& categories_): criteria(criteria_), categories(categories_) {}

  void dump(std::ostream&) const;
  static Domain load(std::istream&);

};

}  // namespace lincs

#endif  // LINCS__IO__DOMAIN_HPP
