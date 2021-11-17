// Copyright 2021 Vincent Jacques

#ifndef IO_HPP_
#define IO_HPP_

#include <vector>
#include <iostream>


namespace ppl::io {

// Classes for easy input/output of domain objects

struct Model {
  Model(
    int criteria_count,
    int categories_count,
    const std::vector<std::vector<float>>& profiles,
    const std::vector<float>& weights);

  bool is_valid() const;

  void save_to(std::ostream&) const;
  static Model load_from(std::istream&);

  const int criteria_count;
  const int categories_count;
  const std::vector<std::vector<float>> profiles;
  const std::vector<float> weights;
};

struct ClassifiedAlternative {
  ClassifiedAlternative(
    const std::vector<float>& criteria_values,
    int assigned_category);

  const std::vector<float> criteria_values;
  int assigned_category;  // @todo Fix this inconsistency: everything but that is const in this module
};

struct LearningSet {
  LearningSet(
    int criteria_count,
    int categories_count,
    int alternatives_count,
    const std::vector<ClassifiedAlternative>& alternatives);

  bool is_valid() const;

  void save_to(std::ostream&) const;
  static LearningSet load_from(std::istream&);

  const int criteria_count;
  const int categories_count;
  const int alternatives_count;
  const std::vector<ClassifiedAlternative> alternatives;
};

}  // namespace ppl::io

#endif  // IO_HPP_
