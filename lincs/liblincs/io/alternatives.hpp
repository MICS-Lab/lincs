// Copyright 2023 Vincent Jacques

#ifndef LINCS__IO__ALTERNATIVES_HPP
#define LINCS__IO__ALTERNATIVES_HPP

#include <map>
#include <optional>
#include <variant>
#include <vector>

#include "problem.hpp"


namespace lincs {

class Performance {
 public:
  static Performance make_real(float value) { return Performance(value); }

  static Performance make_integer(int value) { return Performance(value); }

  static Performance make_enumerated(std::string value) { return Performance(value); }

 private:
  Performance(float value): perf(value) {}

  Performance(int value): perf(value) {}

  Performance(std::string value): perf(value) {}

 public:
  bool operator==(const Performance& other) const { return perf == other.perf; }

 public:
  float get_real_value() const { return std::get<float>(perf); }

  int get_integer_value() const { return std::get<int>(perf); }

  std::string get_enumerated_value() const { return std::get<std::string>(perf); }

 private:
  std::variant<float, int, std::string> perf;
};

class Alternative {
 public:
  Alternative(
    const std::string& name_,
    const std::vector<Performance>& profile_,
    const std::optional<unsigned>& category_index_
  ) :
    name(name_),
    profile(profile_),
    category_index(category_index_)
  {}

 public:
  bool operator==(const Alternative& other) const { return name == other.name && profile == other.profile && category_index == other.category_index; }

 public:
  std::string name;
  std::vector<Performance> profile;
  std::optional<unsigned> category_index;
};

class Alternatives {
 public:
  // @todo(Project management, v1.1) Consider taking 'alternatives_' by rvalue reference and moving it in 'alternatives'
  Alternatives(const Problem&, const std::vector<Alternative>& alternatives_) :
    alternatives(alternatives_)
  {}

 public:
  bool operator==(const Alternatives& other) const {
    return alternatives == other.alternatives;
  }

 public:
  void dump(const Problem&, std::ostream&) const;
  static Alternatives load(const Problem&, std::istream&);

 public:
  std::vector<Alternative> alternatives;
};

}  // namespace lincs

#endif  // LINCS__IO__ALTERNATIVES_HPP
