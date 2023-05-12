// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__ALTERNATIVES_HPP
#define LINCS__IO__ALTERNATIVES_HPP

#include <optional>

#include "domain.hpp"


namespace lincs {

struct Alternative {
  std::string name;
  std::vector<float> profile;
  std::optional<std::string> category;

  Alternative() {}
  Alternative(const std::string& name_, const std::vector<float>& profile_, const std::optional<std::string>& category_): name(name_), profile(profile_), category(category_) {}

  bool operator==(const Alternative& other) const { return name == other.name && profile == other.profile && category == other.category; }
};

struct Alternatives {
  const Domain& domain;
  std::vector<Alternative> alternatives;

  Alternatives(const Domain& domain_, const std::vector<Alternative>& alternatives_): domain(domain_), alternatives(alternatives_) {}

  void dump(std::ostream&) const;
  static Alternatives load(const Domain&, std::istream&);
};

}  // namespace lincs

#endif  // LINCS__IO__ALTERNATIVES_HPP
