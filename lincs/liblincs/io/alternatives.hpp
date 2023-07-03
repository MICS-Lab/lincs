// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__ALTERNATIVES_HPP
#define LINCS__IO__ALTERNATIVES_HPP

#include <optional>

#include "problem.hpp"


namespace lincs {

struct Alternative {
  std::string name;
  std::vector<float> profile;
  std::optional<unsigned> category_index;

  Alternative() {}
  Alternative(const std::string& name_, const std::vector<float>& profile_, const std::optional<unsigned>& category_index_): name(name_), profile(profile_), category_index(category_index_) {}

  bool operator==(const Alternative& other) const { return name == other.name && profile == other.profile && category_index == other.category_index; }
};

struct Alternatives {
  const Problem& problem;
  std::vector<Alternative> alternatives;

  Alternatives(const Problem& problem_, const std::vector<Alternative>& alternatives_): problem(problem_), alternatives(alternatives_) {}

  void dump(const Problem&, std::ostream&) const;
  static Alternatives load(const Problem&, std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__ALTERNATIVES_HPP
