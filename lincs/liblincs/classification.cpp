// Copyright 2023 Vincent Jacques

#include "classification.hpp"

#include <cassert>


namespace lincs {

ClassificationResult classify_alternatives(const Domain& domain, const Model& model, Alternatives* alternatives) {
  assert(&(model.domain) == &domain);
  assert(&(alternatives->domain) == &domain);

  const unsigned criteria_count = domain.criteria.size();
  const unsigned categories_count = domain.categories.size();

  ClassificationResult result{0, 0};

  for (auto& alternative: alternatives->alternatives) {
    uint category_index;
    for (category_index = categories_count - 1; category_index != 0; --category_index) {
      const auto& boundary = model.boundaries[category_index - 1];
      assert(boundary.sufficient_coalitions.kind == Model::SufficientCoalitions::Kind::weights);
      float weight_at_or_above_profile = 0;
      for (uint criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
        const float alternative_value = alternative.profile[criterion_index];
        const float profile_value = boundary.profile[criterion_index];
        if (alternative_value >= profile_value) {
          weight_at_or_above_profile += boundary.sufficient_coalitions.criterion_weights[criterion_index];
        }
      }
      if (weight_at_or_above_profile >= 1.f) {
        break;
      }
    }

    const std::string& category = domain.categories[category_index].name;
    if (alternative.category == category) {
      ++result.unchanged;
    } else {
      alternative.category = category;
      ++result.changed;
    }
  }

  return result;
}

}  // namespace lincs
