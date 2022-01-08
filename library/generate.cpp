// Copyright 2021-2022 Vincent Jacques

#include "generate.hpp"

#include <algorithm>
#include <vector>

#include <chrones.hpp>

#include "assign.hpp"


namespace ppl::generate {

io::Model model(std::mt19937* gen, const uint criteria_count, const uint categories_count) {
  CHRONE();

  // Profile can take any values. We arbitrarily generate them uniformly between 0 and 1
  std::uniform_real_distribution<float> values_distribution(0.0f, 1.0f);

  std::vector<std::vector<float>> profiles(categories_count - 1, std::vector<float>(criteria_count));
  for (uint crit_index = 0; crit_index != criteria_count; ++crit_index) {
    // Profiles must be ordered on each criterion, so we generate a random column...
    std::vector<float> column(categories_count - 1);
    std::generate(
      column.begin(), column.end(),
      [&values_distribution, gen]() { return values_distribution(*gen); });
    // ... sort it...
    std::sort(column.begin(), column.end());
    // ... and assign that column accross all profiles.
    for (uint profile_index = 0; profile_index != categories_count - 1; ++profile_index) {
      profiles[profile_index][crit_index] = column[profile_index];
    }
  }

  // Weights are a bit trickier. Partial sums of weights are compared to 1,
  // so we'll get more meaningful test models if we keep their total sum in that order of magnitude.
  // A model where the sum of all weights is below 1 is useless for test purposes,
  // (because it assigns all alternatives to category 0), so we try to avoid that.
  // It should also be possible to have some huge weights, but not too often.
  // We should get these requirements reviewed by a statistics expert.
  // For the time being, we use a Gamma distribution of shape aplha and scale beta:
  //  - its mean is alpha * beta
  //  - its variance is alpha * beta²
  // Somewhat arbitrarily, we want:
  const float mean = 2. / criteria_count;  // so that the total sum of weights is most often greater than 1
  const float variance = 0.1;  // even more arbitrary
  // So:
  const float alpha = mean * mean / variance;
  const float beta = variance / mean;
  std::gamma_distribution<float> weights_distribution(alpha, beta);

  std::vector<float> weights(criteria_count);
  std::generate(
    weights.begin(), weights.end(),
    [&weights_distribution, gen]() { return weights_distribution(*gen); });

  return io::Model(criteria_count, categories_count, profiles, weights);
}

io::LearningSet learning_set(
    std::mt19937* gen,
    const io::Model& model,
    const uint alternatives_count) {
  CHRONE();

  std::vector<io::ClassifiedAlternative> alternatives;
  alternatives.reserve(alternatives_count);

  // We don't do anything to ensure homogeneous repartition amongst categories.
  // We just generate random profiles uniformly in [0, 1]...
  std::uniform_real_distribution<float> values_distribution(0.0f, 1.0f);

  for (uint alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    std::vector<float> criteria_values(model.criteria_count);
    std::generate(
      criteria_values.begin(), criteria_values.end(),
      [&values_distribution, gen]() { return values_distribution(*gen); });


    alternatives.push_back(io::ClassifiedAlternative(criteria_values, 0/* Temporary assignment */));
  }

  io::LearningSet learning_set(model.criteria_count, model.categories_count, alternatives_count, alternatives);
  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, std::vector<io::Model>(1, model));

  // ... and simulate their category assignment according to "ground truth" (the provided model in that case).
  for (uint alt_index = 0; alt_index != alternatives_count; ++alt_index) {
    alternatives[alt_index].assigned_category = get_assignment(*models, 0, alt_index);
  }

  return io::LearningSet(model.criteria_count, model.categories_count, alternatives_count, alternatives);
}

}  // namespace ppl::generate
