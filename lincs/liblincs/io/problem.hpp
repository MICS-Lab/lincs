// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__IO__PROBLEM_HPP
#define LINCS__IO__PROBLEM_HPP

#include <cassert>
#include <map>
#include <string>
#include <vector>

#include "../variant-dispatch.hpp"
#include "../unreachable.hpp"
#include "validation.hpp"


namespace lincs {

class Criterion {
 public:
  enum class PreferenceDirection {
    increasing,
    isotone=increasing,
    decreasing,
    antitone=decreasing,
    // @todo(Feature, v1.2) Add single-peaked
    // @todo(Feature, later) Add unknown
  };

  class RealValues {
   public:
    RealValues(PreferenceDirection preference_direction_, float min_value_, float max_value_) : preference_direction(preference_direction_), min_value(min_value_), max_value(max_value_) {
      validate(min_value <= max_value, "The min and max values of a real-valued criterion must be ordered.");
    }

   public:
    bool operator==(const RealValues& other) const {
      return preference_direction == other.preference_direction && min_value == other.min_value && max_value == other.max_value;
    }

   public:
    PreferenceDirection get_preference_direction() const { return preference_direction; }

    bool is_increasing() const { return preference_direction == PreferenceDirection::increasing; }

    bool is_decreasing() const { return preference_direction == PreferenceDirection::decreasing; }

    float get_min_value() const { return min_value; }

    float get_max_value() const { return max_value; }

    bool is_acceptable(float value) const { return min_value <= value && value <= max_value; }

   private:
    PreferenceDirection preference_direction;
    float min_value;
    float max_value;
  };

  class IntegerValues {
   public:
    IntegerValues(PreferenceDirection preference_direction_, int min_value_, int max_value_) : preference_direction(preference_direction_), min_value(min_value_), max_value(max_value_) {
      validate(min_value <= max_value, "The min and max values of an integer-valued criterion must be ordered.");
    }

   public:
    bool operator==(const IntegerValues& other) const {
      return preference_direction == other.preference_direction && min_value == other.min_value && max_value == other.max_value;
    }

   public:
    PreferenceDirection get_preference_direction() const { return preference_direction; }

    bool is_increasing() const { return preference_direction == PreferenceDirection::increasing; }

    bool is_decreasing() const { return preference_direction == PreferenceDirection::decreasing; }

    int get_min_value() const { return min_value; }

    int get_max_value() const { return max_value; }

    bool is_acceptable(int value) const { return min_value <= value && value <= max_value; }

   private:
    PreferenceDirection preference_direction;
    int min_value;
    int max_value;
  };

  class EnumeratedValues {
    // @todo(Feature, v1.2) Support single-peaked enumerated criteria

   public:
    EnumeratedValues(const std::vector<std::string>& ordered_values_) : ordered_values(ordered_values_), value_ranks() {
      validate(ordered_values.size() >= 2, "An enumerated criterion must have at least 2 values");
      for (unsigned i = 0; i < ordered_values.size(); ++i) {
        value_ranks[ordered_values[i]] = i;
      }
    }

   public:
    bool operator==(const EnumeratedValues& other) const {
      return ordered_values == other.ordered_values;  // No need to compare value_ranks
    }

   public:
    const std::vector<std::string>& get_ordered_values() const { return ordered_values; }

    const std::map<std::string, unsigned>& get_value_ranks() const { return value_ranks; }

    unsigned get_value_rank(const std::string& value) const { return value_ranks.at(value); }

    bool is_acceptable(const std::string& value) const { return value_ranks.count(value) == 1; }

   private:
    std::vector<std::string> ordered_values;
    std::map<std::string, unsigned> value_ranks;
  };

  // WARNING: keep the enum and the variant consistent
  // (because the variant's index is used as the enum's value)
  enum class ValueType { real, integer, enumerated };
  typedef std::variant<RealValues, IntegerValues, EnumeratedValues> Values;

 public:
  Criterion(const std::string& name_, Values values_) : name(name_), values(values_) {}

 public:
  bool operator==(const Criterion& other) const {
    return name == other.name && values == other.values;
  }

 public:
  const std::string& get_name() const { return name; }

  ValueType get_value_type() const { return ValueType(values.index()); }
  const Values& get_values() const { return values; }

  bool is_real() const { return get_value_type() == ValueType::real; }
  const RealValues& get_real_values() const { return std::get<RealValues>(values); }

  bool is_integer() const { return get_value_type() == ValueType::integer; }
  const IntegerValues& get_integer_values() const { return std::get<IntegerValues>(values); }

  bool is_enumerated() const { return get_value_type() == ValueType::enumerated; }
  const EnumeratedValues& get_enumerated_values() const { return std::get<EnumeratedValues>(values); }

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
  const std::string& get_name() const { return name; }

 private:
  std::string name;
};

class Problem {
 public:
  Problem(const std::vector<Criterion>& criteria_, const std::vector<Category>& ordered_categories_): criteria(criteria_), ordered_categories(ordered_categories_) {
    validate(criteria.size() >= 1, "A problem must have at least one criterion");
    validate(ordered_categories.size() >= 2, "A problem must have at least 2 categories");
  }

 public:
  bool operator==(const Problem& other) const {
    return criteria == other.criteria && ordered_categories == other.ordered_categories;
  }

 public:
  const std::vector<Criterion>& get_criteria() const { return criteria; }
  const std::vector<Category>& get_ordered_categories() const { return ordered_categories; }

 public:
  static const std::string json_schema;
  void dump(std::ostream&) const;
  static Problem load(std::istream&);

 private:
  std::vector<Criterion> criteria;
  std::vector<Category> ordered_categories;
};

}  // namespace lincs

#endif  // LINCS__IO__PROBLEM_HPP
