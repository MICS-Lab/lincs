// Copyright 2021 Vincent Jacques

#include "io.hpp"

#include <algorithm>
#include <numeric>


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

namespace {
  void check_valid(const std::string& type_name, const std::optional<std::string>& error) {
    if (error) {
      std::cerr << "Error during " << type_name << " validation: " << *error << std::endl;
      exit(1);
    }
  }
}

namespace ppl::io {

Model::Model(
  const uint criteria_count_,
  const uint categories_count_,
  const std::vector<std::vector<float>>& profiles_,
  const std::vector<float>& weights_) :
    criteria_count(criteria_count_),
    categories_count(categories_count_),
    profiles(profiles_),
    weights(weights_) {
  check_valid("model", validate());
}

std::optional<std::string> Model::validate() const {
  // Valid counts
  if (criteria_count < 1) return "fewer than 1 criteria";

  if (categories_count < 2) return "fewer than 2 categories";

  // Consistent sizes
  if (profiles.size() != categories_count - 1) return "inconsistent number of profiles";

  if (std::any_of(
    profiles.begin(), profiles.end(),
    [this](const std::vector<float>& profile) { return profile.size() != criteria_count; }
  )) return "inconsistent profile length";

  if (weights.size() != criteria_count) return "inconsistent number of weights";

  // Positive weights...
  if (std::any_of(
    weights.begin(), weights.end(),
    [](float w) { return w < 0; }
  )) return "negative weight";
  // ... with at least one non-zero weight
  if (std::all_of(
    weights.begin(), weights.end(),
    [](float w) { return w == 0; }
  )) return "all weights are zero";

  // Profiles between 0...
  if (std::any_of(
    profiles.front().begin(), profiles.front().end(),
    [](const float v) { return v < 0; }
  )) return "profile below 0.0";
  // ... and 1
  if (std::any_of(
    profiles.back().begin(), profiles.back().end(),
    [](const float v) { return v > 1; }
  )) return "profile above 1.0";

  // Profiles ordered on each criterion, with at least one criterion with different values
  for (uint profile_index = 0; profile_index != categories_count - 2; ++profile_index) {
    bool at_least_one_strictly_above = false;
    for (uint crit_index = 0; crit_index != criteria_count; ++crit_index) {
      const float lower_value = profiles[profile_index][crit_index];
      const float higher_value = profiles[profile_index + 1][crit_index];
      if (higher_value > lower_value) {
        at_least_one_strictly_above = true;
      } else if (higher_value < lower_value) {
        return "pair of unordered profiles";
      }
    }
    if (!at_least_one_strictly_above) return "pair of equal profiles";
  }

  return std::nullopt;
}

void Model::save_to(std::ostream& s) const {
  s << criteria_count << std::endl;
  s << categories_count << std::endl;
  float weights_sum = std::accumulate(weights.begin(), weights.end(), 0.f);
  if (weights_sum == 0) weights_sum = 1;  // Don't crash, just output a model with weights set to zeros
  std::vector<float> normalized_weights(weights);
  std::transform(
    normalized_weights.begin(), normalized_weights.end(),
    normalized_weights.begin(),
    [weights_sum](float w) { return w / weights_sum; });
  s << space_separated(normalized_weights.begin(), normalized_weights.end()) << std::endl;
  s << 1 / weights_sum << std::endl;;
  for (auto profile : profiles) {
    s << space_separated(profile.begin(), profile.end()) << std::endl;
  }
}

Model Model::load_from(std::istream& s) {
  uint criteria_count;
  s >> criteria_count;
  uint categories_count;
  s >> categories_count;
  std::vector<float> weights(criteria_count);
  s >> space_separated(weights.begin(), weights.end());
  float threshold;
  s >> threshold;
  std::transform(weights.begin(), weights.end(), weights.begin(), [threshold](float w) { return w / threshold; });
  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (auto& profile : profiles) {
    s >> space_separated(profile.begin(), profile.end());
  }
  return Model(criteria_count, categories_count, profiles, weights);
}

Model Model::make_homogeneous(uint criteria_count, float weights_sum, uint categories_count) {
  std::vector<std::vector<float>> profiles;
  profiles.reserve(categories_count - 1);
  for (uint profile_index = 0; profile_index != categories_count - 1; ++profile_index) {
    const float value = static_cast<float>(profile_index + 1) / categories_count;
    profiles.push_back(std::vector<float>(criteria_count, value));
  }

  std::vector<float> weights(criteria_count, weights_sum / criteria_count);

  return Model(criteria_count, categories_count, profiles, weights);
}

ClassifiedAlternative::ClassifiedAlternative(
  const std::vector<float>& criteria_values_,
  const uint assigned_category_):
    criteria_values(criteria_values_),  // @todo Rename 'performances'
    assigned_category(assigned_category_) {}

LearningSet::LearningSet(
  const uint criteria_count_,
  const uint categories_count_,
  const uint alternatives_count_,
  const std::vector<ClassifiedAlternative>& alternatives_) :
    criteria_count(criteria_count_),
    categories_count(categories_count_),
    alternatives_count(alternatives_count_),
    alternatives(alternatives_) {
  check_valid("learning set", validate());
}

std::optional<std::string> LearningSet::validate() const {
  // Valid counts
  if (criteria_count < 1) return "fewer than 1 criteria";

  if (categories_count < 2) return "fewer than 2 categories";

  if (alternatives_count < 1) return "fewer than 1 alternatives";

  // Consistent sizes
  if (alternatives.size() != alternatives_count) return "inconsistent number of alternatives";

  if (std::any_of(
    alternatives.begin(), alternatives.end(),
    [this](const ClassifiedAlternative& alt) { return alt.criteria_values.size() != criteria_count; }
  )) return "inconsistent alternative length";

  // Performances between zero and one
  if (std::any_of(
    alternatives.begin(), alternatives.end(),
    [](const ClassifiedAlternative& alt) {
      return std::any_of(
        alt.criteria_values.begin(), alt.criteria_values.end(),
        [](const float performance) { return performance < 0; });
    }
  )) return "performance below 0.0";
  if (std::any_of(
    alternatives.begin(), alternatives.end(),
    [](const ClassifiedAlternative& alt) {
      return std::any_of(
        alt.criteria_values.begin(), alt.criteria_values.end(),
        [](const float performance) { return performance > 1; });
    }
  )) return "performance above 1.0";

  // Assignment less than categories_count
  if (std::any_of(
    alternatives.begin(), alternatives.end(),
    [this](const ClassifiedAlternative& alt) { return alt.assigned_category >= categories_count; }
  )) return "assigned category too large";

  return std::nullopt;
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
  uint criteria_count;
  s >> criteria_count;
  uint categories_count;
  s >> categories_count;
  uint alternatives_count;
  s >> alternatives_count;

  std::vector<ClassifiedAlternative> alternatives;
  alternatives.reserve(alternatives_count);
  for (uint alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(criteria_count);
    s >> space_separated(criteria_values.begin(), criteria_values.end());
    uint assigned_category;
    s >> assigned_category;
    alternatives.push_back(ClassifiedAlternative(criteria_values, assigned_category));
  }

  return LearningSet(criteria_count, categories_count, alternatives_count, alternatives);
}

}  // namespace ppl::io
