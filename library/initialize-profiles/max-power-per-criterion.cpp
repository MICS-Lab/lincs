// Copyright 2021-2022 Vincent Jacques

#include "max-power-per-criterion.hpp"

#include <algorithm>
#include <map>

#include <chrones.hpp>


namespace ppl {

std::map<float, double> get_candidate_probabilities(const DomainView& domain, uint crit_index, uint profile_index) {
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
InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(const Models<Host>& models) {
  CHRONE();

  ModelsView models_view = models.get_view();

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
  RandomNumberGenerator random,
  Models<Host>* models,
  const uint iteration_index,
  std::vector<uint>::const_iterator model_indexes_begin,
  const std::vector<uint>::const_iterator model_indexes_end
) {
  CHRONE();

  ModelsView models_view = models->get_view();

  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const uint model_index = *model_indexes_begin;

    models_view.initialization_iteration_indexes[model_index] = iteration_index;

    // Embarrassingly parallel
    for (uint crit_index = 0; crit_index != models_view.domain.criteria_count; ++crit_index) {
      // Not parallel because of the profiles ordering constraint
      for (uint category_index = models_view.domain.categories_count - 1; category_index != 0; --category_index) {
        const uint profile_index = category_index - 1;
        float value = _generators[crit_index][profile_index](random.urbg());

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
