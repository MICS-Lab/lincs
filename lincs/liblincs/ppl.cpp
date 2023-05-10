#include "ppl.hpp"
#line 1 "io.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "io.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <sstream>

#include <chrones.hpp>


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
  CHRONE();

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

std::string encode_floats(float f1, float f2) {
  std::ostringstream oss;
  oss << f1 << ":" << std::hexfloat << f2;
  return oss.str();
}

std::string encode_float(float f) {
  return encode_floats(f, f);
}

std::pair<float, float> decode_floats(const std::string& s) {
  std::istringstream iss(s);
  float f1;
  char c = 0;
  std::string s2;
  float f2 = std::nan("");
  // Not using hexfloat here: see https://stackoverflow.com/a/42617165/905845
  iss >> f1 >> c >> s2;
  if (c == ':') {
    f2 = std::strtof(s2.c_str(), NULL);
  }
  return std::make_pair(f1, f2);
}

float decode_float(const std::string& s) {
  const auto [f1, f2] = decode_floats(s);

  if (encode_float(f2) == s) {
    return f2;
  } else {
    return f1;
  }
}

float encode_weights(const std::vector<float>& denormalized_weights, std::ostream& s) {
  const int criteria_count = denormalized_weights.size();

  float weights_sum = std::accumulate(denormalized_weights.begin(), denormalized_weights.end(), 0.f);
  if (weights_sum == 0) weights_sum = 1;  // Don't crash, just output a model with weights set to zeros

  std::vector<float> normalized_weights(criteria_count);
  std::transform(
    denormalized_weights.begin(), denormalized_weights.end(),
    normalized_weights.begin(),
    [weights_sum](float w) { return w / weights_sum; });

  std::vector<std::string> encoded_weights(criteria_count);
  std::transform(
    normalized_weights.begin(), normalized_weights.end(),
    denormalized_weights.begin(),
    encoded_weights.begin(),
    encode_floats);
  s << space_separated(encoded_weights.begin(), encoded_weights.end());

  return 1 / weights_sum;
}

}  // namespace

void Model::save_to(std::ostream& s) const {
  CHRONE();

  s << criteria_count << std::endl;
  s << categories_count << std::endl;

  const float threshold = encode_weights(weights, s);
  s << std::endl;
  s << threshold << std::endl;  // No need to encode_float:
  // threshold from file is not used if hex-encoded non-normalized weights are used

  for (auto profile : profiles) {
    std::vector<std::string> encoded_profile(criteria_count);
    std::transform(profile.begin(), profile.end(), encoded_profile.begin(), encode_float);
    s << space_separated(encoded_profile.begin(), encoded_profile.end()) << std::endl;
  }
}

Model Model::load_from(std::istream& s) {
  CHRONE();

  uint criteria_count;
  s >> criteria_count;

  uint categories_count;
  s >> categories_count;

  std::string encoded_weights_s;
  std::getline(s, encoded_weights_s);  // Skip the end of previous line
  std::getline(s, encoded_weights_s);

  float threshold;
  s >> threshold;

  std::vector<float> denormalized_weights(criteria_count);
  {
    std::vector<float> normalized_weights(criteria_count);

    std::istringstream iss(encoded_weights_s);

    std::vector<std::string> encoded_weights(criteria_count);
    iss >> space_separated(encoded_weights.begin(), encoded_weights.end());

    std::vector<std::pair<float, float>> decoded_weights(criteria_count);
    std::transform(encoded_weights.begin(), encoded_weights.end(), decoded_weights.begin(), decode_floats);

    std::transform(
      decoded_weights.begin(), decoded_weights.end(),
      normalized_weights.begin(),
      [](std::pair<float, float> p) { return p.first; });

    std::transform(
      decoded_weights.begin(), decoded_weights.end(),
      denormalized_weights.begin(),
      [](std::pair<float, float> p) { return p.second; });

    std::ostringstream reencoded_weights;
    encode_weights(denormalized_weights, reencoded_weights);

    if (reencoded_weights.str() != encoded_weights_s) {
      std::transform(
        normalized_weights.begin(), normalized_weights.end(),
        denormalized_weights.begin(),
        [threshold](float w) { return w / threshold; });
    }
  }

  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (auto& profile : profiles) {
    std::vector<std::string> encoded_profile(criteria_count);
    s >> space_separated(encoded_profile.begin(), encoded_profile.end());
    std::transform(encoded_profile.begin(), encoded_profile.end(), profile.begin(), decode_float);
  }

  return Model(criteria_count, categories_count, profiles, denormalized_weights);
}

Model Model::make_homogeneous(uint criteria_count, float weights_sum, uint categories_count) {
  CHRONE();

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
  CHRONE();

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
  CHRONE();

  s << criteria_count << std::endl;
  s << categories_count << std::endl;
  s << alternatives_count << std::endl;
  for (auto alternative : alternatives) {
    std::vector<std::string> encoded_values(criteria_count);
    std::transform(
      alternative.criteria_values.begin(), alternative.criteria_values.end(),
      encoded_values.begin(),
      encode_float);
    s << space_separated(encoded_values.begin(), encoded_values.end())
      << " " << alternative.assigned_category << std::endl;
  }
}

LearningSet LearningSet::load_from(std::istream& s) {
  CHRONE();

  uint criteria_count;
  s >> criteria_count;
  uint categories_count;
  s >> categories_count;
  uint alternatives_count;
  s >> alternatives_count;

  std::vector<ClassifiedAlternative> alternatives;
  alternatives.reserve(alternatives_count);
  for (uint alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<std::string> encoded_values(criteria_count);
    s >> space_separated(encoded_values.begin(), encoded_values.end());
    std::vector<float> criteria_values(criteria_count);
    std::transform(encoded_values.begin(), encoded_values.end(), criteria_values.begin(), decode_float);
    uint assigned_category;
    s >> assigned_category;
    alternatives.push_back(ClassifiedAlternative(criteria_values, assigned_category));
  }

  return LearningSet(criteria_count, categories_count, alternatives_count, alternatives);
}

}  // namespace ppl::io
#line 1 "problem.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "problem.hpp"

#include <algorithm>
#include <set>
#include <utility>

#include <chrones.hpp>

#include "io.hpp"


namespace ppl {

template<typename Space>
Domain<Space>::Domain(
  Privacy,
  const uint categories_count,
  const uint criteria_count,
  const uint learning_alternatives_count,
  Array2D<Space, float>&& learning_alternatives,
  Array1D<Space, uint>&& learning_assignments) :
    _categories_count(categories_count),
    _criteria_count(criteria_count),
    _learning_alternatives_count(learning_alternatives_count),
    _learning_alternatives(std::move(learning_alternatives)),
    _learning_assignments(std::move(learning_assignments)) {}

template<>
std::shared_ptr<Domain<Host>> Domain<Host>::make(const io::LearningSet& learning_set) {
  CHRONE();

  assert(learning_set.is_valid());

  Array2D<Host, float> alternatives(learning_set.criteria_count, learning_set.alternatives_count, uninitialized);
  Array1D<Host, uint> assignments(learning_set.alternatives_count, uninitialized);

  for (uint alt_index = 0; alt_index != learning_set.alternatives_count; ++alt_index) {
    const io::ClassifiedAlternative& alt = learning_set.alternatives[alt_index];

    for (uint crit_index = 0; crit_index != learning_set.criteria_count; ++crit_index) {
      alternatives[crit_index][alt_index] = alt.criteria_values[crit_index];
    }

    assignments[alt_index] = alt.assigned_category;
  }

  return std::make_shared<Domain<Host>>(
    Privacy(),
    learning_set.categories_count,
    learning_set.criteria_count,
    learning_set.alternatives_count,
    std::move(alternatives),
    std::move(assignments));
}

template<typename Space>
DomainView<Space> Domain<Space>::get_view() const {
  return {
    _categories_count,
    _criteria_count,
    _learning_alternatives_count,
    ArrayView2D<Space, const float>(_learning_alternatives),
    ArrayView1D<Space, const uint>(_learning_assignments),
  };
}

template class Domain<Host>;
template class Domain<Device>;

template<typename Space>
Models<Space>::Models(
  Privacy,
  std::shared_ptr<Domain<Space>> domain,
  const uint models_count,
  Array1D<Space, uint>&& initialization_iteration_indexes,
  Array2D<Space, float>&& weights,
  Array3D<Space, float>&& profiles) :
    _domain(domain),
    _models_count(models_count),
    _initialization_iteration_indexes(std::move(initialization_iteration_indexes)),
    _weights(std::move(weights)),
    _profiles(std::move(profiles)) {}

template<>
std::shared_ptr<Models<Host>> Models<Host>::make(std::shared_ptr<Domain<Host>> domain, const uint models_count) {
  CHRONE();

  DomainView<Host> domain_view = domain->get_view();

  Array1D<Host, uint> initialization_iteration_indexes(models_count, uninitialized);
  Array2D<Host, float> weights(domain_view.criteria_count, models_count, uninitialized);
  Array3D<Host, float> profiles(
    domain_view.criteria_count, (domain_view.categories_count - 1), models_count, uninitialized);

  return std::make_shared<Models>(
    Privacy(),
    domain,
    models_count,
    std::move(initialization_iteration_indexes),
    std::move(weights),
    std::move(profiles));
}

template<>
std::shared_ptr<Models<Host>> Models<Host>::make(
  std::shared_ptr<Domain<Host>> domain,
  const std::vector<io::Model>& models
) {
  CHRONE();

  const uint models_count = models.size();
  auto r = make(domain, models_count);
  auto view = r->get_view();

  for (uint model_index = 0; model_index != models_count; ++model_index) {
    const io::Model& model = models[model_index];
    assert(model.is_valid());

    for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
      view.weights[crit_index][model_index] = model.weights[crit_index];
    }

    for (uint cat_index = 0; cat_index != view.domain.categories_count - 1; ++cat_index) {
      const std::vector<float>& category_profile = model.profiles[cat_index];
      for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
        view.profiles[crit_index][cat_index][model_index] = category_profile[crit_index];
      }
    }
  }

  return r;
}

template<>
io::Model Models<Host>::unmake_one(uint model_index) const {
  CHRONE();

  ModelsView view = get_view();

  std::vector<std::vector<float>> profiles(view.domain.categories_count - 1);
  for (uint cat_index = 0; cat_index != view.domain.categories_count - 1; ++cat_index) {
    profiles[cat_index].reserve(view.domain.criteria_count);
    for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
      profiles[cat_index].push_back(view.profiles[crit_index][cat_index][model_index]);
    }
  }

  std::vector<float> weights;
  weights.reserve(view.domain.criteria_count);
  for (uint crit_index = 0; crit_index != view.domain.criteria_count; ++crit_index) {
    weights.push_back(view.weights[crit_index][model_index]);
  }

  return io::Model(view.domain.criteria_count, view.domain.categories_count, profiles, weights);
}

template<>
std::vector<io::Model> Models<Host>::unmake() const {
  CHRONE();

  ModelsView<Host> view = get_view();

  std::vector<io::Model> models;

  models.reserve(view.models_count);
  for (uint model_index = 0; model_index != _models_count; ++model_index) {
    models.push_back(unmake_one(model_index));
  }

  return models;
}

template<typename Space>
ModelsView<Space> Models<Space>::get_view() const {
  DomainView domain = _domain->get_view();
  return {
    domain,
    _models_count,
    ref(_initialization_iteration_indexes),
    ref(_weights),
    ref(_profiles),
  };
}

template class Models<Host>;
template class Models<Device>;

}  // namespace ppl
#line 1 "assign.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef ASSIGN_HPP_
#define ASSIGN_HPP_

#include "problem.hpp"


namespace ppl {

__host__ __device__
uint get_assignment(const ModelsView<Anywhere>&, uint model_index, uint alternative_index);
uint get_assignment(const Models<Host>&, uint model_index, uint alternative_index);

// Accuracy is returned as an integer between `0` and `models.domain.alternatives_count`.
// (To get the accuracy described in the thesis, it should be divided by `models.domain.alternatives_count`)
uint get_accuracy(const Models<Host>&, uint model_index);
// @todo Remove: this is used only by tests
uint get_accuracy(const Models<Device>&, uint model_index);

}  // namespace ppl

#endif  // ASSIGN_HPP_
#line 1 "assign.cu"
// Copyright 2021-2022 Vincent Jacques

#include "assign.hpp"

#include <chrones.hpp>
#include <lov-e.hpp>


namespace ppl {

__host__ __device__
uint get_assignment(const ModelsView<Anywhere>& models, const uint model_index, const uint alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index < models.models_count);
  assert(alternative_index < models.domain.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (uint category_index = models.domain.categories_count - 1; category_index != 0; --category_index) {
    const uint profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alternative_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

uint get_assignment(const Models<Host>& models, const uint model_index, const uint alternative_index) {
  return get_assignment(models.get_view(), model_index, alternative_index);
}

__host__ __device__
bool is_correctly_assigned(
    const ModelsView<Anywhere>& models,
    const uint model_index,
    const uint alternative_index) {
  const uint expected_assignment = models.domain.learning_assignments[alternative_index];
  const uint actual_assignment = get_assignment(models, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

uint get_accuracy(const Models<Host>& models, const uint model_index) {
  uint accuracy = 0;

  ModelsView<Anywhere> models_view = models.get_view();
  for (uint alt_index = 0; alt_index != models_view.domain.learning_alternatives_count; ++alt_index) {
    if (is_correctly_assigned(models_view, model_index, alt_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

namespace {

typedef GridFactory1D<512> grid;

}  // namespace

__global__ void get_accuracy__kernel(ModelsView<Anywhere> models, const uint model_index, uint* const accuracy) {
  const uint alt_index = grid::x();
  assert(alt_index < models.domain.learning_alternatives_count + grid::blockDim.x);

  if (alt_index < models.domain.learning_alternatives_count) {
    if (is_correctly_assigned(models, model_index, alt_index)) {
      atomicInc(accuracy, models.domain.learning_alternatives_count);
    }
  }
}

uint get_accuracy(const Models<Device>& models, const uint model_index) {
  CHRONE();

  uint* device_accuracy = Device::alloc<uint>(1);
  cudaMemset(device_accuracy, 0, sizeof(uint));
  check_last_cuda_error_no_sync();

  ModelsView<Anywhere> models_view = models.get_view();
  Grid grid = grid::make(models_view.domain.learning_alternatives_count);
  get_accuracy__kernel<<<LOVE_CONFIG(grid)>>>(models_view, model_index, device_accuracy);
  check_last_cuda_error_sync_device();

  uint host_accuracy;
  From<Device>::To<Host>::copy(1, device_accuracy, &host_accuracy);  // NOLINT(build/include_what_you_use)
  Device::free(device_accuracy);
  return host_accuracy;
}

}  // namespace ppl
#line 1 "observe/report-progress.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OBSERVE_REPORT_PROGRESS_HPP_
#define OBSERVE_REPORT_PROGRESS_HPP_


#include <iostream>

#include "../observe.hpp"

namespace ppl {

class ReportProgress : public LearningObserver {
 public:
  void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>& models) override {
    std::cerr << "After iteration n°" << iteration_index << ": best accuracy = " <<
      best_accuracy << "/" << models.get_view().domain.learning_alternatives_count << std::endl;
  }
};

}  // namespace ppl

#endif  // OBSERVE_REPORT_PROGRESS_HPP_
#line 1 "observe/dump-intermediate-models.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_
#define OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_

#include <iostream>

#include "../observe.hpp"


namespace ppl {

class DumpIntermediateModels : public LearningObserver {
 public:
  explicit DumpIntermediateModels(std::ostream& stream_);

  void after_main_iteration(int iteration_index, int best_accuracy, const Models<Host>&) override;

 private:
  std::ostream& stream;
};

}  // namespace ppl

#endif  // OBSERVE_DUMP_INTERMEDIATE_MODELS_HPP_
#line 1 "observe/dump-intermediate-models.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "dump-intermediate-models.hpp"

#include "../assign.hpp"


namespace ppl {

DumpIntermediateModels::DumpIntermediateModels(std::ostream& stream_) :
    stream(stream_) {
  // Emitting YAML by hand... we could do better, but it works for now
  stream << "iterations:" << std::endl;
}

void DumpIntermediateModels::after_main_iteration(int iteration_index, int, const Models<Host>& models) {
  stream
    << "  - iteration_index: " << iteration_index << "\n"
    << "    models:\n";

  auto models_view = models.get_view();

  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    stream
      << "      - model_index: " << model_index << "\n"
      << "        initialization_iteration_index: " << models_view.initialization_iteration_indexes[model_index] << "\n"
      << "        accuracy: " << get_accuracy(models, model_index) << "\n"
      << "        profiles:\n";
    for (uint profile_index = 0; profile_index != models_view.domain.categories_count - 1; ++profile_index) {
      stream << "         - [";
      for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
        if (crit_index != 0) stream << ", ";
        stream << models_view.profiles[crit_index][profile_index][model_index];
      }
      stream << "]\n";
    }
    stream
      << "        weights: [";
      for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
        if (crit_index != 0) stream << ", ";
        stream << models_view.weights[crit_index][model_index];
      }
    stream << "]\n";
  }

  stream << std::flush;
}

}  // namespace ppl
#line 1 "initialize-profiles/max-power-per-criterion.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_
#define INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_

#include <memory>
#include <vector>

#include "../initialize-profiles.hpp"
#include "../randomness.hpp"


namespace ppl {

/*
Implement 3.3.2 of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(const Random&, const Models<Host>&);

  void initialize_profiles(
    std::shared_ptr<Models<Host>>,
    uint iteration_index,
    std::vector<uint>::const_iterator model_indexes_begin,
    std::vector<uint>::const_iterator model_indexes_end) override;

 private:
  const Random& _random;
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> _generators;
};

// class InitializeProfilesForDeterministicMaximalDiscriminationPowerPerCriterion :
// public ProfilesInitializationStrategy {
//  public:
//   void initialize_profiles(
//     Models<Host>* models,
//     uint iteration_index,
//     std::vector<uint>::const_iterator model_indexes_begin,
//     std::vector<uint>::const_iterator model_indexes_end) override;
// };

}  // namespace ppl

#endif  // INITIALIZE_PROFILES_MAX_POWER_PER_CRITERION_HPP_
#line 1 "initialize-profiles/max-power-per-criterion.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "max-power-per-criterion.hpp"

#include <algorithm>
#include <map>

#include <chrones.hpp>


namespace ppl {

std::map<float, double> get_candidate_probabilities(
  const DomainView<Host>& domain,
  uint crit_index,
  uint profile_index
) {
  CHRONE();

  std::vector<float> values_below;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
  values_below.reserve(domain.learning_alternatives_count);
  std::vector<float> values_above;
  values_above.reserve(domain.learning_alternatives_count);
  // This loop could/should be done once outside this function
  for (uint alt_index = 0; alt_index != domain.learning_alternatives_count; ++alt_index) {
    const float value = domain.learning_alternatives[crit_index][alt_index];
    const uint assignment = domain.learning_assignments[alt_index];
    if (assignment == profile_index) {
      values_below.push_back(value);
    } else if (assignment == profile_index + 1) {
      values_above.push_back(value);
    }
  }

  std::map<float, double> candidate_probabilities;

  for (auto candidates : { values_below, values_above }) {
    for (auto candidate : candidates) {
      if (candidate_probabilities.find(candidate) != candidate_probabilities.end()) {
        // Candidate value has already been evaluated (because it appears several times)
        continue;
      }

      uint correctly_classified_count = 0;
      // @todo Could we somehow sort 'values_below' and 'values_above' and walk the values only once?
      // (Transforming this O(n²) loop in O(n*log n) + O(n))
      for (auto value : values_below) if (value < candidate) ++correctly_classified_count;
      for (auto value : values_above) if (value >= candidate) ++correctly_classified_count;
      candidate_probabilities[candidate] = static_cast<double>(correctly_classified_count) / candidates.size();
    }
  }

  return candidate_probabilities;
}

InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::
InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(
  const Random& random,
  const Models<Host>& models) :
    _random(random) {
  CHRONE();

  ModelsView<Host> models_view = models.get_view();

  _generators.reserve(models_view.domain.categories_count - 1);

  for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
    _generators.push_back(std::vector<ProbabilityWeightedGenerator<float>>());
    _generators.back().reserve(models_view.domain.criteria_count);
    for (uint profile_index = 0; profile_index != models_view.domain.categories_count - 1; ++profile_index) {
      _generators.back().push_back(ProbabilityWeightedGenerator<float>::make(
        get_candidate_probabilities(models_view.domain, crit_index, profile_index)));
    }
  }
}

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  std::shared_ptr<Models<Host>> models,
  const uint iteration_index,
  std::vector<uint>::const_iterator model_indexes_begin,
  const std::vector<uint>::const_iterator model_indexes_end
) {
  CHRONE();

  ModelsView<Host> models_view = models->get_view();

  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const uint model_index = *model_indexes_begin;

    models_view.initialization_iteration_indexes[model_index] = iteration_index;

    // Embarrassingly parallel
    for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
      // Not parallel because of the profiles ordering constraint
      for (uint category_index = models_view.domain.categories_count - 1; category_index != 0; --category_index) {
        const uint profile_index = category_index - 1;
        float value = _generators[crit_index][profile_index](_random.urbg());

        if (profile_index != models_view.domain.categories_count - 2) {
          value = std::min(value, models_view.profiles[crit_index][profile_index + 1][model_index]);
        }
        // @todo Add a unit test that triggers the following assertion
        // (This will require removing the code to enforce the order of profiles above)
        // Then restore the code to enforce the order of profiles
        // Note, this assertion does not protect us from initializing a model with two identical profiles.
        // Is it really that bad?
        assert(
          profile_index == models_view.domain.categories_count - 2
          || models_view.profiles[crit_index][profile_index + 1][model_index] >= value);

        models_view.profiles[crit_index][profile_index][model_index] = value;
      }
    }
  }
}

}  // namespace ppl
#line 1 "improve-profiles/desirability.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_DESIRABILITY_HPP_
#define IMPROVE_PROFILES_DESIRABILITY_HPP_

#include "../problem.hpp"


namespace ppl {

struct Desirability {
  // Value for moves with no impact.
  // @todo Verify with Vincent Mousseau that this is the correct value.
  static constexpr float zero_value = 0;

  uint v = 0;
  uint w = 0;
  uint q = 0;
  uint r = 0;
  uint t = 0;

  __host__ __device__
  float value() const {
    if (v + w + t + q + r == 0) {
      return zero_value;
    } else {
      return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
    }
  }
};

__host__ __device__
void update_move_desirability(
  const ModelsView<Anywhere>& models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination,
  const uint alt_index,
  Desirability* desirability
);

}  // namespace ppl


template<>
inline ppl::Desirability* Host::alloc<ppl::Desirability>(const std::size_t n) {
  return Host::force_alloc<ppl::Desirability>(n);
}

template<>
inline void Host::memset<ppl::Desirability>(const std::size_t n, const char v, ppl::Desirability* const p) {
  Host::force_memset<ppl::Desirability>(n, v, p);
}

template<>
inline ppl::Desirability* Device::alloc<ppl::Desirability>(const std::size_t n) {
  return Device::force_alloc<ppl::Desirability>(n);
}

template<>
inline void Device::memset<ppl::Desirability>(const std::size_t n, const char v, ppl::Desirability* const p) {
  Device::force_memset<ppl::Desirability>(n, v, p);
}

#endif  // IMPROVE_PROFILES_DESIRABILITY_HPP_
#line 1 "improve-profiles/desirability.cu"
// Copyright 2021-2022 Vincent Jacques

#include "desirability.hpp"

#include "../assign.hpp"


namespace ppl {

__host__ __device__
void increment(
    uint* i,
    uint
    #ifdef __CUDA_ARCH__
    max
    #endif
) {
  #ifdef __CUDA_ARCH__
  atomicInc(i, max);
  #else
  ++*i;
  #endif
}

__host__ __device__
void update_move_desirability(
  const ModelsView<Anywhere>& models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination,
  const uint alt_index,
  Desirability* desirability
) {
  const float current_position = models.profiles[criterion_index][profile_index][model_index];
  const float weight = models.weights[criterion_index][model_index];

  const float value = models.domain.learning_alternatives[criterion_index][alt_index];
  const uint learning_assignment = models.domain.learning_assignments[alt_index];
  const uint model_assignment = get_assignment(models, model_index, alt_index);

  // @todo Factorize with get_assignment
  float weight_at_or_above_profile = 0;
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
    const float profile_value = models.profiles[crit_index][profile_index][model_index];
    if (alternative_value >= profile_value) {
      weight_at_or_above_profile += models.weights[crit_index][model_index];
    }
  }

  // These imbricated conditionals could be factorized, but this form has the benefit
  // of being a direct translation of the top of page 78 of Sobrie's thesis.
  // Correspondance:
  // - learning_assignment: bottom index of A*
  // - model_assignment: top index of A*
  // - profile_index: h
  // - destination: b_j +/- \delta
  // - current_position: b_j
  // - value: a_j
  // - weight_at_or_above_profile: \sigma
  // - weight: w_j
  // - 1: \lambda
  if (destination > current_position) {
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight < 1) {
        increment(&(desirability->v), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight >= 1) {
        increment(&(desirability->w), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index + 1
      && destination > value
      && value >= current_position
      && weight_at_or_above_profile - weight < 1) {
        increment(&(desirability->q), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination > value
      && value >= current_position) {
        increment(&(desirability->r), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment < profile_index
      && model_assignment > profile_index
      && destination > value
      && value >= current_position) {
        increment(&(desirability->t), models.domain.learning_alternatives_count);
    }
  } else {
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        increment(&(desirability->v), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index + 1
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight < 1) {
        increment(&(desirability->w), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index
      && destination < value
      && value < current_position
      && weight_at_or_above_profile + weight >= 1) {
        increment(&(desirability->q), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment == profile_index
      && model_assignment == profile_index + 1
      && destination <= value
      && value < current_position) {
        increment(&(desirability->r), models.domain.learning_alternatives_count);
    }
    if (
      learning_assignment > profile_index + 1
      && model_assignment < profile_index + 1
      && destination < value
      && value <= current_position) {
        increment(&(desirability->t), models.domain.learning_alternatives_count);
    }
  }
}

}  // namespace ppl
#line 1 "improve-profiles/accuracy-heuristic-cpu.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_ACCURACY_HEURISTIC_CPU_HPP_
#define IMPROVE_PROFILES_ACCURACY_HEURISTIC_CPU_HPP_

#include <memory>

#include <chrones.hpp>

#include "../improve-profiles.hpp"
#include "../randomness.hpp"


namespace ppl {

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ImproveProfilesWithAccuracyHeuristicOnCpu : public ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnCpu(const Random& random) : _random(random) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  const Random& _random;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_ACCURACY_HEURISTIC_CPU_HPP_
#line 1 "improve-profiles/accuracy-heuristic-cpu.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "accuracy-heuristic-cpu.hpp"

#include <algorithm>
#include <utility>
#include <cassert>
#include <random>

#include <chrones.hpp>

#include "../assign.hpp"
#include "desirability.hpp"

namespace ppl {

namespace {

Desirability compute_move_desirability(
  const ModelsView<Host>& models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index,
  const float destination
) {
  Desirability d;

  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    update_move_desirability(
      models, model_index, profile_index, criterion_index, destination, alt_index, &d);
  }

  return d;
}

void improve_model_profile(
  const Random& random,
  ModelsView<Host> models,
  const uint model_index,
  const uint profile_index,
  const uint criterion_index
) {
  CHRONE();

  // WARNING: We're assuming all criteria have values in [0, 1]
  // @todo Can we relax this assumption?
  // This is consistent with our comment in the header file, but slightly less generic than Sobrie's thesis
  const float lowest_destination =
    profile_index == 0 ? 0. :
    models.profiles[criterion_index][profile_index - 1][model_index];
  const float highest_destination =
    profile_index == models.domain.categories_count - 2 ? 1. :
    models.profiles[criterion_index][profile_index + 1][model_index];

  float best_destination = models.profiles[criterion_index][profile_index][model_index];
  float best_desirability = Desirability().value();

  if (lowest_destination == highest_destination) {
    assert(best_destination == lowest_destination);
    return;
  }

  // Not sure about this part: we're considering an arbitrary number of possible moves as described in
  // Mousseau's prez-mics-2018(8).pdf, but:
  //  - this is wasteful when there are fewer alternatives in the interval
  //  - this is not strictly consistent with, albeit much simpler than, Sobrie's thesis
  // @todo Ask Vincent Mousseau about the following:
  // We could consider only a finite set of values for b_j described as follows:
  // - sort all the 'a_j's
  // - compute all midpoints between two successive 'a_j'
  // - add two extreme values (0 and 1, or above the greatest a_j and below the smallest a_j)
  // Then instead of taking a random values in [lowest_destination, highest_destination],
  // we'd take a random subset of the intersection of these midpoints with that interval.
  for (uint n = 0; n < 64; ++n) {
    // Map (embarrassingly parallel)
    const float destination = random.uniform_float(lowest_destination, highest_destination);
    const float desirability = compute_move_desirability(
      models, model_index, profile_index, criterion_index, destination).value();
    // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  // @todo Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (random.uniform_float(0, 1) <= best_desirability) {
    models.profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

void improve_model_profile(
  const Random& random,
  ModelsView<Host> models,
  const uint model_index,
  const uint profile_index,
  ArrayView1D<Anywhere, const uint> criterion_indexes
) {
  CHRONE();

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    improve_model_profile(random, models, model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

template<typename T>
void swap(T& a, T& b) {
  T c = a;
  a = b;
  b = c;
}

template<typename T>
void shuffle(const Random& random, ArrayView1D<Anywhere, T> m) {
  for (uint i = 0; i != m.s0(); ++i) {
    swap(m[i], m[random.uniform_int(0, m.s0())]);
  }
}

void improve_model_profiles(const Random& random, const ModelsView<Host>& models, const uint model_index) {
  CHRONE();

  Array1D<Host, uint> criterion_indexes(models.domain.criteria_count, uninitialized);
  // Not worth parallelizing because models.domain.criteria_count is typically small
  for (uint crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (uint profile_index = 0; profile_index != models.domain.categories_count - 1; ++profile_index) {
    shuffle<uint>(random, ref(criterion_indexes));
    improve_model_profile(random, models, model_index, profile_index, criterion_indexes);
  }
}

}  // namespace

void ImproveProfilesWithAccuracyHeuristicOnCpu::improve_profiles(std::shared_ptr<Models<Host>> models) {
  CHRONE();

  auto models_view = models->get_view();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    improve_model_profiles(_random, models_view, model_index);
  }
}

}  // namespace ppl
#line 1 "optimize-weights/glop.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_GLOP_HPP_
#define OPTIMIZE_WEIGHTS_GLOP_HPP_

#include <memory>
#include <vector>

#include "../optimize-weights.hpp"


namespace ppl {

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
using GLOP to solve the linear program.
*/
class OptimizeWeightsUsingGlop : public WeightsOptimizationStrategy {
 public:
  struct LinearProgram;

 public:
  void optimize_weights(std::shared_ptr<Models<Host>>) override;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_GLOP_HPP_
#line 1 "optimize-weights/glop.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "glop.hpp"

#include <ortools/glop/lp_solver.h>

#include <string>
#include <vector>
#include <memory>

#include <chrones.hpp>


namespace ppl {

namespace glp = operations_research::glop;

struct OptimizeWeightsUsingGlop::LinearProgram {
  std::shared_ptr<glp::LinearProgram> program;
  std::vector<glp::ColIndex> weight_variables;
  std::vector<glp::ColIndex> x_variables;
  std::vector<glp::ColIndex> xp_variables;
  std::vector<glp::ColIndex> y_variables;
  std::vector<glp::ColIndex> yp_variables;
};

std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> make_internal_linear_program(
  const float epsilon,
  const ModelsView<Host>& models,
  uint model_index
) {
  CHRONE();

  auto lp = std::make_shared<OptimizeWeightsUsingGlop::LinearProgram>();

  lp->program = std::make_shared<glp::LinearProgram>();
  lp->weight_variables.reserve(models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->weight_variables.push_back(lp->program->CreateNewVariable());
  }

  lp->x_variables.reserve(models.domain.learning_alternatives_count);
  lp->xp_variables.reserve(models.domain.learning_alternatives_count);
  lp->y_variables.reserve(models.domain.learning_alternatives_count);
  lp->yp_variables.reserve(models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->x_variables.push_back(lp->program->CreateNewVariable());
    lp->xp_variables.push_back(lp->program->CreateNewVariable());
    lp->y_variables.push_back(lp->program->CreateNewVariable());
    lp->yp_variables.push_back(lp->program->CreateNewVariable());

    lp->program->SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program->SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1, 1);
      lp->program->SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program->SetCoefficient(c, lp->xp_variables.back(), 1);
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
        }
      }
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program->SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program->SetCoefficient(c, lp->yp_variables.back(), -1);
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
        }
      }
    }
  }

  return lp;
}

std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> make_verbose_linear_program(
    const float epsilon, const ModelsView<Host>& models, uint model_index) {
  CHRONE();

  auto lp = make_internal_linear_program(epsilon, models, model_index);

  assert(lp->weight_variables.size() == models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->program->SetVariableName(lp->weight_variables[crit_index], "w_" + std::to_string(crit_index));
  }

  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->program->SetVariableName(lp->x_variables[alt_index], "x_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->xp_variables[alt_index], "x'_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->y_variables[alt_index], "y_" + std::to_string(alt_index));
    lp->program->SetVariableName(lp->yp_variables[alt_index], "y'_" + std::to_string(alt_index));
  }

  return lp;
}

std::shared_ptr<glp::LinearProgram> make_verbose_linear_program(
    const float epsilon, std::shared_ptr<Models<Host>> models_, uint model_index) {
  CHRONE();

  return make_verbose_linear_program(epsilon, models_->get_view(), model_index)->program;
}

auto solve_linear_program(std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> lp) {
  CHRONE();

  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  auto status = solver.Solve(*lp->program);
  assert(status == operations_research::glop::ProblemStatus::OPTIMAL);
  auto values = solver.variable_values();

  return values;
}

void optimize_weights(const ModelsView<Host>& models, uint model_index) {
  CHRONE();

  auto lp = make_internal_linear_program(1e-6, models, model_index);
  auto values = solve_linear_program(lp);

  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    models.weights[crit_index][model_index] = values[lp->weight_variables[crit_index]];
  }
}

void optimize_weights(const ModelsView<Host>& models) {
  CHRONE();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models.models_count; ++model_index) {
    optimize_weights(models, model_index);
  }
}

void OptimizeWeightsUsingGlop::optimize_weights(std::shared_ptr<Models<Host>> models) {
  CHRONE();

  ppl::optimize_weights(models->get_view());
}

}  // namespace ppl
#line 1 "optimize-weights/glop-reuse.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_
#define OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_

#include <ortools/glop/lp_solver.h>

#include <memory>
#include <vector>

#include "../optimize-weights.hpp"


namespace ppl {

/*
Implement 3.3.3 of https://tel.archives-ouvertes.fr/tel-01370555/document
using GLOP to solve the linear program, reusing the linear programs and solvers
to try and benefit from the reuse optimization in GLOP.
*/
class OptimizeWeightsUsingGlopAndReusingPrograms : public WeightsOptimizationStrategy {
 public:
  explicit OptimizeWeightsUsingGlopAndReusingPrograms(const Models<Host>&);

 public:
  void optimize_weights(std::shared_ptr<Models<Host>>) override;

 public:
  struct LinearProgram {
    operations_research::glop::LinearProgram program;

    std::vector<operations_research::glop::ColIndex> weight_variables;
    std::vector<operations_research::glop::ColIndex> x_variables;
    std::vector<operations_research::glop::ColIndex> xp_variables;
    std::vector<operations_research::glop::ColIndex> y_variables;
    std::vector<operations_research::glop::ColIndex> yp_variables;

    // @todo Fix naming
    std::vector<operations_research::glop::RowIndex> a_constraints;
    std::vector<operations_research::glop::RowIndex> b_constraints;
  };

 private:
  std::vector<LinearProgram> _linear_programs;
  std::vector<operations_research::glop::LPSolver> _solvers;
};

}  // namespace ppl

#endif  // OPTIMIZE_WEIGHTS_GLOP_REUSE_HPP_
#line 1 "optimize-weights/glop-reuse.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "glop-reuse.hpp"

#include <string>
#include <vector>

#include <chrones.hpp>


namespace ppl {

namespace glp = operations_research::glop;

void structure_linear_program(
  OptimizeWeightsUsingGlopAndReusingPrograms::LinearProgram* lp,
  const float epsilon,
  const ModelsView<Host>& models
) {
  lp->weight_variables.reserve(models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->weight_variables.push_back(lp->program.CreateNewVariable());
  }

  lp->x_variables.reserve(models.domain.learning_alternatives_count);
  lp->xp_variables.reserve(models.domain.learning_alternatives_count);
  lp->y_variables.reserve(models.domain.learning_alternatives_count);
  lp->yp_variables.reserve(models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->x_variables.push_back(lp->program.CreateNewVariable());
    lp->xp_variables.push_back(lp->program.CreateNewVariable());
    lp->y_variables.push_back(lp->program.CreateNewVariable());
    lp->yp_variables.push_back(lp->program.CreateNewVariable());

    lp->program.SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program.SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program.CreateNewConstraint();
      lp->a_constraints.push_back(c);
      lp->program.SetConstraintBounds(c, 1, 1);
      lp->program.SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program.SetCoefficient(c, lp->xp_variables.back(), 1);
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->program.CreateNewConstraint();
      lp->b_constraints.push_back(c);
      lp->program.SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program.SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program.SetCoefficient(c, lp->yp_variables.back(), -1);
    }
  }
}

void label_linear_program(
  OptimizeWeightsUsingGlopAndReusingPrograms::LinearProgram* lp,
  const ModelsView<Host>& models
) {
  assert(lp->weight_variables.size() == models.domain.criteria_count);
  for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
    lp->program.SetVariableName(lp->weight_variables[crit_index], "w_" + std::to_string(crit_index));
  }

  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);
  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    lp->program.SetVariableName(lp->x_variables[alt_index], "x_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->xp_variables[alt_index], "x'_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->y_variables[alt_index], "y_" + std::to_string(alt_index));
    lp->program.SetVariableName(lp->yp_variables[alt_index], "y'_" + std::to_string(alt_index));
  }
}

void update_linear_program(
  OptimizeWeightsUsingGlopAndReusingPrograms::LinearProgram* lp,
  const ModelsView<Host>& models,
  const uint model_index
) {
  assert(lp->weight_variables.size() == models.domain.criteria_count);
  assert(lp->x_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->xp_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->y_variables.size() == models.domain.learning_alternatives_count);
  assert(lp->yp_variables.size() == models.domain.learning_alternatives_count);

  uint a_index = 0;
  uint b_index = 0;

  for (uint alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    const uint category_index = models.domain.learning_assignments[alt_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->a_constraints[a_index++];
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 1);
        } else {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 0);
        }
      }
    }

    if (category_index != models.domain.categories_count - 1) {
      glp::RowIndex c = lp->b_constraints[b_index++];
      for (uint crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
        const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
        const float profile_value = models.profiles[crit_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 1);
        } else {
          lp->program.SetCoefficient(c, lp->weight_variables[crit_index], 0);
        }
      }
    }
  }
}

std::shared_ptr<glp::LinearProgram> make_verbose_linear_program_reuse(
    const float epsilon, std::shared_ptr<Models<Host>> models, uint model_index) {
  auto models_view = models->get_view();

  OptimizeWeightsUsingGlopAndReusingPrograms::LinearProgram lp;
  structure_linear_program(&lp, epsilon, models_view);
  label_linear_program(&lp, models_view);
  update_linear_program(&lp, models_view, model_index);

  auto r = std::make_shared<glp::LinearProgram>();
  r->PopulateFromLinearProgram(lp.program);
  return r;
}

OptimizeWeightsUsingGlopAndReusingPrograms::OptimizeWeightsUsingGlopAndReusingPrograms(
  const ppl::Models<Host>& models
) :
    _linear_programs(models.get_view().models_count),
    _solvers(models.get_view().models_count) {
  CHRONE();

  #pragma omp parallel for
  for (auto& lp : _linear_programs) {
    structure_linear_program(&lp, 1e-6, models.get_view());
  }

  glp::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  #pragma omp parallel for
  for (auto& solver : _solvers) {
    solver.SetParameters(parameters);
  }
}

void OptimizeWeightsUsingGlopAndReusingPrograms::optimize_weights(std::shared_ptr<Models<Host>> models) {
  CHRONE();

  auto models_view = models->get_view();

  #pragma omp parallel for
  for (uint model_index = 0; model_index != models_view.models_count; ++model_index) {
    OptimizeWeightsUsingGlopAndReusingPrograms::LinearProgram& lp = _linear_programs[model_index];
    update_linear_program(&lp, models_view, model_index);
    lp.program.CleanUp();

    auto status = _solvers[model_index].Solve(lp.program);
    assert(status == glp::ProblemStatus::OPTIMAL);
    auto values = _solvers[model_index].variable_values();

    for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
      models_view.weights[crit_index][model_index] = values[lp.weight_variables[crit_index]];
    }
  }
}

}  // namespace ppl
#line 1 "terminate/accuracy.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ACCURACY_HPP_
#define TERMINATE_ACCURACY_HPP_

#include "../terminate.hpp"


namespace ppl {

class TerminateAtAccuracy : public TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(uint target_accuracy) :
    _target_accuracy(target_accuracy) {}

  bool terminate(uint /*iteration_index*/, uint best_accuracy) override {
    return best_accuracy >= _target_accuracy;
  }

 private:
  uint _target_accuracy;
};

}  // namespace ppl

#endif  // TERMINATE_ACCURACY_HPP_
#line 1 "terminate/iterations.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ITERATIONS_HPP_
#define TERMINATE_ITERATIONS_HPP_

#include "../terminate.hpp"


namespace ppl {

class TerminateAfterIterations : public TerminationStrategy {
 public:
  explicit TerminateAfterIterations(uint max_iterations) :
    _max_iterations(max_iterations) {}

  bool terminate(uint iteration_index, uint /*best_accuracy*/) override {
    return iteration_index >= _max_iterations;
  }

 private:
  uint _max_iterations;
};

}  // namespace ppl

#endif  // TERMINATE_ITERATIONS_HPP_
#line 1 "terminate/duration.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_DURATION_HPP_
#define TERMINATE_DURATION_HPP_

#include <chrono>  // NOLINT(build/c++11)

#include "../terminate.hpp"


namespace ppl {

class TerminateAfterDuration : public TerminationStrategy {
  typedef std::chrono::steady_clock clock;

 public:
  explicit TerminateAfterDuration(typename clock::duration max_duration) :
    _max_time(clock::now() + max_duration) {}

  bool terminate(uint /*iteration_index*/, uint /*best_accuracy*/) override {
    return clock::now() >= _max_time;
  }

 private:
  typename clock::time_point _max_time;
};

}  // namespace ppl

#endif  // TERMINATE_DURATION_HPP_
#line 1 "terminate/any.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef TERMINATE_ANY_HPP_
#define TERMINATE_ANY_HPP_

#include <memory>
#include <vector>

#include "../terminate.hpp"


namespace ppl {

class TerminateOnAny : public TerminationStrategy {
 public:
  explicit TerminateOnAny(std::vector<std::shared_ptr<TerminationStrategy>> strategies) :
    _strategies(strategies) {}

  bool terminate(uint iteration_index, uint best_accuracy) override {
    for (auto strategy : _strategies) {
      if (strategy->terminate(iteration_index, best_accuracy)) {
        return true;
      }
    }

    return false;
  }

 private:
  std::vector<std::shared_ptr<TerminationStrategy>> _strategies;
};

}  // namespace ppl

#endif  // TERMINATE_ANY_HPP_
#line 1 "median-and-max.hpp"
// Copyright 2021-2022 Vincent Jacques

#ifndef MEDIAN_AND_MAX_HPP_
#define MEDIAN_AND_MAX_HPP_

#include <algorithm>

/*
Ensure that the median and maximum values of the range [begin, end[ are
in the correct positions (middle and last).
Also ensure that all values below the median are before the median,
and all values above the median are after the median.
*/
template<typename RandomIt, class Compare>
void ensure_median_and_max(RandomIt begin, RandomIt end, Compare comp) {
  auto len = end - begin;
  if (len == 0) return;
  // Ensure max
  std::nth_element(begin, begin + len - 1, end, comp);
  // Ensure median, not touching max
  std::nth_element(begin, begin + len / 2, begin + len - 1, comp);
}

#endif  // MEDIAN_AND_MAX_HPP_
#line 1 "learning.cpp"
// Copyright 2021-2022 Vincent Jacques

#include "learning.hpp"

#include <algorithm>
#include <numeric>

#include <chrones.hpp>

#include "assign.hpp"
#include "median-and-max.hpp"


namespace ppl {

std::vector<uint> partition_models_by_accuracy(const uint models_count, const Models<Host>& models) {
  CHRONE();

  std::vector<uint> accuracies(models_count, 0);
  for (uint model_index = 0; model_index != models_count; ++model_index) {
    accuracies[model_index] = get_accuracy(models, model_index);
  }

  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [&accuracies](uint left_model_index, uint right_model_index) {
      return accuracies[left_model_index] < accuracies[right_model_index];
    });

  return model_indexes;
}

LearningResult perform_learning(
  std::shared_ptr<Models<Host>> models,
  std::vector<std::shared_ptr<LearningObserver>> observers,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
) {
  CHRONE();

  const uint models_count = models->get_view().models_count;

  std::vector<uint> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  profiles_initialization_strategy->initialize_profiles(
    models,
    0,
    model_indexes.begin(), model_indexes.end());

  uint best_accuracy = 0;

  for (int iteration_index = 0; !termination_strategy->terminate(iteration_index, best_accuracy); ++iteration_index) {
    if (iteration_index != 0) {
      profiles_initialization_strategy->initialize_profiles(
        models,
        iteration_index,
        model_indexes.begin(), model_indexes.begin() + models_count / 2);
    }

    weights_optimization_strategy->optimize_weights(models);
    profiles_improvement_strategy->improve_profiles(models);

    model_indexes = partition_models_by_accuracy(models_count, *models);
    best_accuracy = get_accuracy(*models, model_indexes.back());

    for (auto observer : observers) {
      observer->after_main_iteration(iteration_index, best_accuracy, *models);
    }
  }

  return LearningResult(models->unmake_one(model_indexes.back()), best_accuracy);
}

}  // namespace ppl
#line 1 "tools/learn.cpp"
// Copyright 2021-2022 Vincent Jacques

#include <algorithm>
#include <chrono>  // NOLINT(build/c++11)
#include <fstream>
#include <iostream>

#include <chrones.hpp>
#include <CLI11.hpp>
#include <magic_enum.hpp>

#include "../library/improve-profiles/accuracy-heuristic-cpu.hpp"
#include "../library/improve-profiles/accuracy-heuristic-gpu.hpp"
#include "../library/initialize-profiles/max-power-per-criterion.hpp"
#include "../library/learning.hpp"
#include "../library/observe/dump-intermediate-models.hpp"
#include "../library/observe/report-progress.hpp"
#include "../library/optimize-weights/glop.hpp"
#include "../library/optimize-weights/glop-reuse.hpp"
#include "../library/terminate/accuracy.hpp"
#include "../library/terminate/any.hpp"
#include "../library/terminate/duration.hpp"
#include "../library/terminate/iterations.hpp"


CHRONABLE("learn")

std::vector<std::shared_ptr<ppl::LearningObserver>> make_observers(
  const bool quiet,
  std::optional<std::ofstream>& intermediate_models_file
) {
  std::vector<std::shared_ptr<ppl::LearningObserver>> observers;

  if (intermediate_models_file) {
    observers.push_back(std::make_shared<ppl::DumpIntermediateModels>(*intermediate_models_file));
  }

  if (!quiet) {
    observers.push_back(std::make_shared<ppl::ReportProgress>());
  }

  return observers;
}

std::shared_ptr<ppl::ProfilesInitializationStrategy> make_profiles_initialization_strategy(
  const Random& random,
  const ppl::Models<Host>& models
) {
  // @todo Complete with other strategies
  return std::make_shared<ppl::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion>(
    random, models);
}

enum class WeightsOptimizationStrategy {
  glop,
  glop_reuse,
};

std::shared_ptr<ppl::WeightsOptimizationStrategy> make_weights_optimization_strategy(
  WeightsOptimizationStrategy strategy,
  const ppl::Models<Host>& host_models
) {
  switch (strategy) {
    case WeightsOptimizationStrategy::glop:
      return std::make_shared<ppl::OptimizeWeightsUsingGlop>();
    case WeightsOptimizationStrategy::glop_reuse:
      return std::make_shared<ppl::OptimizeWeightsUsingGlopAndReusingPrograms>(host_models);
  }
  throw std::runtime_error("Unknown weights optimization strategy");
}

enum class ProfilesImprovementStrategy {
  heuristic,
};

std::shared_ptr<ppl::ProfilesImprovementStrategy> make_profiles_improvement_strategy(
  ProfilesImprovementStrategy strategy,
  const bool use_gpu,
  const Random& random,
  std::shared_ptr<ppl::Domain<Host>> domain,
  std::shared_ptr<ppl::Models<Host>> models
) {
  switch (strategy) {
    case ProfilesImprovementStrategy::heuristic:
      if (use_gpu) {
        return std::make_shared<ppl::ImproveProfilesWithAccuracyHeuristicOnGpu>(
          random, models->clone_to<Device>(domain->clone_to<Device>()));
      } else {
        return std::make_shared<ppl::ImproveProfilesWithAccuracyHeuristicOnCpu>(random);
      }
  }
  throw std::runtime_error("Unknown profiles improvement strategy");
}

std::shared_ptr<ppl::TerminationStrategy> make_termination_strategy(
  uint target_accuracy,
  std::optional<uint> max_iterations,
  std::optional<std::chrono::seconds> max_duration
) {
  std::vector<std::shared_ptr<ppl::TerminationStrategy>> termination_strategies;

  termination_strategies.push_back(
    std::make_shared<ppl::TerminateAtAccuracy>(target_accuracy));

  if (max_iterations) {
    termination_strategies.push_back(
      std::make_shared<ppl::TerminateAfterIterations>(*max_iterations));
  }

  if (max_duration) {
    termination_strategies.push_back(
      std::make_shared<ppl::TerminateAfterDuration>(*max_duration));
  }

  if (termination_strategies.size() == 1) {
    return termination_strategies[0];
  } else {
    return std::make_shared<ppl::TerminateOnAny>(termination_strategies);
  }
}
