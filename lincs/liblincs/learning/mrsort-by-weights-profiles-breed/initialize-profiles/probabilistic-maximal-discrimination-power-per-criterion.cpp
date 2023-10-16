// Copyright 2023 Vincent Jacques

#include "probabilistic-maximal-discrimination-power-per-criterion.hpp"

#include "../../../chrones.hpp"
#include "../../../generation.hpp"  // Only for tests

#include "../../../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(LearningData& learning_data_) : learning_data(learning_data_) {
  CHRONE();

  generators.reserve(learning_data.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
    auto& generator = generators.emplace_back();
    generator.reserve(learning_data.categories_count - 1);
    for (unsigned profile_index = 0; profile_index != learning_data.categories_count - 1; ++profile_index) {
      generator.emplace_back(get_candidate_probabilities(criterion_index, profile_index));
    }
  }
}

std::map<float, double> InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities(
  unsigned criterion_index,
  unsigned profile_index
) {
  CHRONE();

  const Criterion& criterion = learning_data.problem.criteria[criterion_index];

  std::vector<float> values_worse;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
  values_worse.reserve(learning_data.learning_alternatives_count);
  std::vector<float> values_better;
  values_better.reserve(learning_data.learning_alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != learning_data.learning_alternatives_count; ++alternative_index) {
    const float value = learning_data.learning_alternatives[criterion_index][alternative_index];
    const unsigned assignment = learning_data.learning_assignments[alternative_index];
    if (assignment == profile_index) {
      values_worse.push_back(value);
    } else if (assignment == profile_index + 1) {
      values_better.push_back(value);
    }
  }

  if (values_better.empty() && values_worse.empty()) {
    return {{criterion.min_value, 1.0}};
  } else {
    std::map<float, double> candidate_probabilities;

    for (auto candidates : { values_worse, values_better }) {
      for (auto candidate : candidates) {
        if (candidate_probabilities.find(candidate) != candidate_probabilities.end()) {
          // Candidate value has already been evaluated (because it appears several times)
          continue;
        }

        unsigned correctly_classified_count = 0;
        // @todo(Performance, later) Could we somehow sort 'values_worse' and 'values_better' and walk the values only once?
        // (Transforming this O(n²) loop in O(n*log n) + O(n))
        for (auto value : values_worse) if (criterion.strictly_better(candidate, value)) ++correctly_classified_count;
        for (auto value : values_better) if (criterion.better_or_equal(value, candidate)) ++correctly_classified_count;
        candidate_probabilities[candidate] = static_cast<double>(correctly_classified_count) / candidates.size();
      }
    }

    assert(!candidate_probabilities.empty());
    return candidate_probabilities;
  }
}

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  unsigned model_indexes_begin,
  const unsigned model_indexes_end
) {
  CHRONE();

  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const unsigned model_index = learning_data.model_indexes[model_indexes_begin];

    // Embarrassingly parallel
    for (unsigned criterion_index = 0; criterion_index != learning_data.criteria_count; ++criterion_index) {
      const Criterion& criterion = learning_data.problem.criteria[criterion_index];
      // Not parallel because of the profiles ordering constraint
      for (unsigned category_index = learning_data.categories_count - 1; category_index != 0; --category_index) {
        const unsigned profile_index = category_index - 1;
        float value = generators[criterion_index][profile_index](learning_data.urbgs[model_index]);

        // Enforce profiles ordering constraint
        if (criterion.category_correlation == Criterion::CategoryCorrelation::growing) {
          if (profile_index != learning_data.categories_count - 2) {
            value = std::min(value, learning_data.profiles[criterion_index][profile_index + 1][model_index]);
          }
        } else {
          assert(criterion.category_correlation == Criterion::CategoryCorrelation::decreasing);
          if (profile_index != learning_data.categories_count - 2) {
            value = std::max(value, learning_data.profiles[criterion_index][profile_index + 1][model_index]);
          }
        }

        learning_data.profiles[criterion_index][profile_index][model_index] = value;
      }
    }
  }
}

TEST_CASE("Initialize profiles - respect ordering") {
  Problem problem{
    {
      Criterion(
        "Criterion 1",
        Criterion::ValueType::real,
        Criterion::CategoryCorrelation::growing,
        0.0, 1.0
      ),
      Criterion(
        "Criterion 2",
        Criterion::ValueType::real,
        Criterion::CategoryCorrelation::decreasing,
        0.0, 1.0
      )
    },
    {
      Category("Category 1"),
      Category("Category 2"),
      Category("Category 3"),
    }
  };
  Model model = generate_mrsort_classification_model(problem, 42);
  auto learning_set = generate_classified_alternatives(problem, model, 1000, 42, 0.1);
  auto learning_data = LearnMrsortByWeightsProfilesBreed::LearningData::make(problem, learning_set, 1, 42);
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion initializer(learning_data);

  for (unsigned iteration = 0; iteration != 10; ++iteration) {
    initializer.initialize_profiles(0, 1);
    // Both CHECKs fail at least once when the 'Enforce profiles ordering constraint' code is removed
    CHECK(learning_data.profiles[0][0][0] <= learning_data.profiles[0][1][0]);
    CHECK(learning_data.profiles[1][0][0] >= learning_data.profiles[1][1][0]);
  }
}


}  // namespace lincs
