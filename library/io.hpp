// Copyright 2021 Vincent Jacques

#ifndef IO_HPP_
#define IO_HPP_

#include <vector>
#include <iostream>


namespace ppl::io {

typedef unsigned int uint;

// Classes for easy input/output of domain objects

struct Model {
  Model(
    uint criteria_count,
    uint categories_count,
    const std::vector<std::vector<float>>& profiles,
    const std::vector<float>& weights);

  bool is_valid() const;

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

  bool is_valid() const;

  void save_to(std::ostream&) const;
  static LearningSet load_from(std::istream&);

  const uint criteria_count;
  const uint categories_count;
  const uint alternatives_count;
  std::vector<ClassifiedAlternative> alternatives;
};

}  // namespace ppl::io

#endif  // IO_HPP_
