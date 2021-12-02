// Copyright 2021 Vincent Jacques

#ifndef IO_HPP_
#define IO_HPP_

#include <iostream>
#include <optional>
#include <string>
#include <vector>


namespace ppl::io {

// Classes for easy input/output of domain objects

struct Model {
  Model(
    uint criteria_count,
    uint categories_count,
    const std::vector<std::vector<float>>& profiles,
    // @todo Add min/max values for each criterion.
    // Then remove (everywhere) the assumption that these values are 0 and 1.
    const std::vector<float>& weights);

  std::optional<std::string> validate() const;
  bool is_valid() const { return !validate(); }

  void save_to(std::ostream&) const;
  static Model load_from(std::istream&);
  static Model make_homogeneous(uint criteria_count, float weights_sum, uint categories_count);

  const uint criteria_count;
  const uint categories_count;
  std::vector<std::vector<float>> profiles;
  std::vector<float> weights;
};

struct ClassifiedAlternative {
  ClassifiedAlternative(
    const std::vector<float>& criteria_values,
    uint assigned_category);

  const std::vector<float> criteria_values;
  uint assigned_category;  // @todo Fix this inconsistency: everything but that is const in this module
};

struct LearningSet {
  LearningSet(
    uint criteria_count,
    uint categories_count,
    uint alternatives_count,
    const std::vector<ClassifiedAlternative>& alternatives);

  std::optional<std::string> validate() const;
  bool is_valid() const { return !validate(); }

  void save_to(std::ostream&) const;
  static LearningSet load_from(std::istream&);

  const uint criteria_count;
  const uint categories_count;
  const uint alternatives_count;
  std::vector<ClassifiedAlternative> alternatives;
};

}  // namespace ppl::io

#endif  // IO_HPP_
