// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__IO__ALTERNATIVES_HPP
#define LINCS__IO__ALTERNATIVES_HPP

#include <map>
#include <optional>
#include <variant>
#include <vector>

#include "../internal.hpp"
#include "problem.hpp"


namespace lincs {

class Performance {
 public:
  class Real {
   public:
    Real(float value_) : value(value_) {}

   public:
    bool operator==(const Real& other) const { return value == other.value; }

   public:
    float get_value() const { return value; }

   private:
    float value;
  };

  class Integer {
   public:
    Integer(int value_) : value(value_) {}

   public:
    bool operator==(const Integer& other) const { return value == other.value; }

   public:
    int get_value() const { return value; }

   private:
    int value;
  };

  class Enumerated {
   public:
    Enumerated(std::string value_) : value(value_) {}

   public:
    bool operator==(const Enumerated& other) const { return value == other.value; }

   public:
    std::string get_value() const { return value; }

   private:
    std::string value;
  };

  typedef std::variant<Real, Integer, Enumerated> Self;

 public:
  Performance(const Self& self_) : self(self_) {}

  // Copyable and movable
  Performance(const Performance&) = default;
  Performance& operator=(const Performance&) = default;
  Performance(Performance&&) = default;
  Performance& operator=(Performance&&) = default;

 public:
  bool operator==(const Performance& other) const {
    return self == other.self;
  }

 public:
  Criterion::ValueType get_value_type() const { return Criterion::ValueType(self.index()); }
  const Self& get() const { return self; }

  bool is_real() const { return get_value_type() == Criterion::ValueType::real; }
  Real get_real() const { return std::get<Real>(self); }

  bool is_integer() const { return get_value_type() == Criterion::ValueType::integer; }
  Integer get_integer() const { return std::get<Integer>(self); }

  bool is_enumerated() const { return get_value_type() == Criterion::ValueType::enumerated; }
  Enumerated get_enumerated() const { return std::get<Enumerated>(self); }

 private:
  Self self;
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
  bool operator==(const Alternative& other) const {
    return name == other.name && profile == other.profile && category_index == other.category_index;
  }

 public:
  const std::string& get_name() const { return name; }
  void set_name(const std::string& name_) { name = name_; }

  const std::vector<Performance>& get_profile() const { return profile; }

  const std::optional<unsigned>& get_category_index() const { return category_index; }
  void set_category_index(const std::optional<unsigned>& category_index_) { category_index = category_index_; }

 private:
  std::string name;
  std::vector<Performance> profile;
  std::optional<unsigned> category_index;
};

class Alternatives {
 public:
  Alternatives(const Problem&, const std::vector<Alternative>&);

  Alternatives(Internal, const std::vector<Alternative>& alternatives_) : alternatives(alternatives_) {}

 public:
  bool operator==(const Alternatives& other) const {
    return alternatives == other.alternatives;
  }

 public:
  void dump(const Problem&, std::ostream&) const;
  static Alternatives load(const Problem&, std::istream&);

 public:
  const std::vector<Alternative>& get_alternatives() const { return alternatives; }
  std::vector<Alternative>& get_writable_alternatives() { return alternatives; }

 private:
  std::vector<Alternative> alternatives;
};

}  // namespace lincs

#endif  // LINCS__IO__ALTERNATIVES_HPP
