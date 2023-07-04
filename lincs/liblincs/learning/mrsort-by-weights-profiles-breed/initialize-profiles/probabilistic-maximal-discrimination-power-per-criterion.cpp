// Copyright 2023 Vincent Jacques

#include "probabilistic-maximal-discrimination-power-per-criterion.hpp"


namespace lincs {

InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(LearningData& learning_data_) : learning_data(learning_data_) {
  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    generators.emplace_back();
    for (unsigned profile_index = 0; profile_index != learning_data.categories_count - 1; ++profile_index) {
      generators.back().emplace_back(get_candidate_probabilities(criterion_index, profile_index));
    }
  }
}

std::map<float, double> InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities(
  unsigned criterion_index,
  unsigned profile_index
) {
  std::vector<float> values_below;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
  values_below.reserve(learning_data.learning_alternatives_count);
  std::vector<float> values_above;
  values_above.reserve(learning_data.learning_alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    const float value = learning_data.learning_alternatives[criterion_index][alternative_index];
    const unsigned assignment = learning_data.learning_assignments[alternative_index];
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

      unsigned correctly_classified_count = 0;
      // @todo Could we somehow sort 'values_below' and 'values_above' and walk the values only once?
      // (Transforming this O(n²) loop in O(n*log n) + O(n))
      for (auto value : values_below) if (value < candidate) ++correctly_classified_count;
      for (auto value : values_above) if (value >= candidate) ++correctly_classified_count;
      candidate_probabilities[candidate] = static_cast<double>(correctly_classified_count) / candidates.size();
    }
  }

  return candidate_probabilities;
}

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const unsigned model_index = learning_data.model_indexes[model_indexes_begin];

    // Embarrassingly parallel
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      // Not parallel because of the profiles ordering constraint
      for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
        const unsigned profile_index = category_index - 1;
        float value = generators[criterion_index][profile_index](learning_data.urbgs[model_index]);

        if (profile_index != learning_data.categories_count - 2) {
          value = std::min(value, learning_data.profiles[criterion_index][profile_index + 1][model_index]);
        }
        // @todo Add a unit test that triggers the following assertion
        // (This will require removing the code to enforce the order of profiles above)
        // Then restore the code to enforce the order of profiles
        // Note, this assertion does not protect us from initializing a model with two identical profiles.
        // Is it really that bad?
        assert(
          profile_index == learning_data.categories_count - 2
          || learning_data.profiles[criterion_index][profile_index + 1][model_index] >= value);

        learning_data.profiles[criterion_index][profile_index][model_index] = value;
      }
    }
  }
}

}  // namespace lincs
