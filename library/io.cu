// Copyright 2021 Vincent Jacques

#include "io.hpp"

#include <cassert>
#include <algorithm>


namespace {

template<typename T>
struct space_separated {
  space_separated(T begin_, const T& end_) : begin(begin_), end(end_) {}

  T begin;
  const T end;
};

template<typename T>
std::ostream& operator<<(std::ostream& s, space_separated<T> v) {
  if (v.begin != v.end) {
    s << *v.begin;
    while (++v.begin != v.end) {
      s << " " << *v.begin;
    }
  }
  return s;
}

template<typename T>
std::istream& operator>>(std::istream& s, space_separated<T> v) {
  if (v.begin != v.end) {
    s >> *v.begin;
    while (++v.begin != v.end) {
      s >> *v.begin;
    }
  }
  return s;
}

}  // namespace

namespace ppl::io {

Model::Model(
  const int criteria_count_,
  const int categories_count_,
  const std::vector<std::vector<float>>& profiles_,
  const std::vector<float>& weights_) :
    criteria_count(criteria_count_),
    categories_count(categories_count_),
    profiles(profiles_),
    weights(weights_) {
  assert(is_valid());
}

bool Model::is_valid() const {
  // Valid counts
  if (criteria_count < 1) return false;

  if (categories_count < 2) return false;

  // Consistent sizes
  if (profiles.size() != categories_count - 1) return false;

  if (std::any_of(
      profiles.begin(), profiles.end(),
      [this](const std::vector<float>& profile) { return profile.size() != criteria_count; }
  )) return false;

  if (weights.size() != criteria_count) return false;

  // Positive weights
  if (std::any_of(
    weights.begin(), weights.end(),
    [](float w) { return w < 0; }
  )) return false;

  // @todo Profiles between 0 and 1
  // @todo Profiles ordered on each criterion

  return true;
}

void Model::save_to(std::ostream& s) const {
  s << criteria_count << std::endl;
  s << categories_count << std::endl;
  s << space_separated(weights.begin(), weights.end()) << std::endl;
  for (auto profile : profiles) {
    s << space_separated(profile.begin(), profile.end()) << std::endl;
  }
}

Model Model::load_from(std::istream& s) {
  int criteria_count;
  s >> criteria_count;
  int categories_count;
  s >> categories_count;
  std::vector<float> weights(criteria_count);
  s >> space_separated(weights.begin(), weights.end());
  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (auto& profile : profiles) {
    s >> space_separated(profile.begin(), profile.end());
  }
  return Model(criteria_count, categories_count, profiles, weights);
}

Model Model::make_homogeneous(int criteria_count, float weights_sum, int categories_count) {
  std::vector<std::vector<float>> profiles;
  profiles.reserve(categories_count - 1);
  for (int profile_index = 0; profile_index != categories_count - 1; ++profile_index) {
    const float value = static_cast<float>(profile_index + 1) / categories_count;
    profiles.push_back(std::vector<float>(criteria_count, value));
  }

  std::vector<float> weights(criteria_count, weights_sum / criteria_count);

  return Model(criteria_count, categories_count, profiles, weights);
}

ClassifiedAlternative::ClassifiedAlternative(
  const std::vector<float>& criteria_values_,
  const int assigned_category_):
    criteria_values(criteria_values_),
    assigned_category(assigned_category_) {}

LearningSet::LearningSet(
  const int criteria_count_,
  const int categories_count_,
  const int alternatives_count_,
  const std::vector<ClassifiedAlternative>& alternatives_) :
    criteria_count(criteria_count_),
    categories_count(categories_count_),
    alternatives_count(alternatives_count_),
    alternatives(alternatives_) {
  assert(is_valid());
}

bool LearningSet::is_valid() const {
  // @todo Check everything
  return true;
}

void LearningSet::save_to(std::ostream& s) const {
  s << criteria_count << std::endl;
  s << categories_count << std::endl;
  s << alternatives_count << std::endl;
  for (auto alternative : alternatives) {
    s << space_separated(alternative.criteria_values.begin(), alternative.criteria_values.end())
      << " " << alternative.assigned_category << std::endl;
  }
}

LearningSet LearningSet::load_from(std::istream& s) {
  int criteria_count;
  s >> criteria_count;
  int categories_count;
  s >> categories_count;
  int alternatives_count;
  s >> alternatives_count;

  std::vector<ClassifiedAlternative> alternatives;
  alternatives.reserve(alternatives_count);
  for (int alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(criteria_count);
    s >> space_separated(criteria_values.begin(), criteria_values.end());
    int assigned_category;
    s >> assigned_category;
    alternatives.push_back(ClassifiedAlternative(criteria_values, assigned_category));
  }

  return LearningSet(criteria_count, categories_count, alternatives_count, alternatives);
}

}  // namespace ppl::io
