// Copyright 2023 Vincent Jacques

#ifndef LINCS__GENERATION_HPP
#define LINCS__GENERATION_HPP

#include <vector>

#include "io.hpp"


namespace lincs {

// @todo(Feature, later) Use structures to hold generation options?
// This would allow calls like:
// generate_classification_problem(4, 3, 42, {.normalized_min_max = true}})
// generate_mrsort_classification_model(problem, 42, {.fixed_weights_sum = 2.0f}})

Problem generate_classification_problem(
  unsigned criteria_count, unsigned categories_count, unsigned random_seed, bool normalized_min_max=true, bool allow_decreasing_criteria=false);

Model generate_mrsort_classification_model(
  const Problem&, unsigned random_seed, std::optional<float> fixed_weights_sum = std::nullopt);

class BalancedAlternativesGenerationException : public std::exception {
 public:
  explicit BalancedAlternativesGenerationException(const std::vector<unsigned>& histogram_) : histogram(histogram_) {}

  const char* what() const noexcept override {
    return "Unable to generate balanced alternatives. Try increasing the allowed imbalance, or use a more lenient model?";
  }

  std::vector<unsigned> histogram;
};

Alternatives generate_classified_alternatives(
  const Problem&,
  const Model&,
  unsigned alternatives_count,
  unsigned random_seed,
  std::optional<float> max_imbalance = std::nullopt
);

void misclassify_alternatives(const Problem&, Alternatives*, unsigned misclassification_count, unsigned random_seed);

} // namespace lincs

#endif  // LINCS__GENERATION_HPP
