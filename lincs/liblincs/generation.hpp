// Copyright 2023 Vincent Jacques

#ifndef LINCS__GENERATION_HPP
#define LINCS__GENERATION_HPP

#include <map>

#include "io.hpp"


namespace lincs {

Problem generate_problem(unsigned criteria_count, unsigned categories_count, unsigned random_seed);

Model generate_mrsort_model(const Problem&, unsigned random_seed, std::optional<float> fixed_weights_sum = std::nullopt);

class BalancedAlternativesGenerationException : public std::exception {
 public:
  explicit BalancedAlternativesGenerationException(const std::map<std::string, unsigned>& histogram_) : histogram(histogram_) {}

  const char* what() const noexcept override {
    return "Unable to generate balanced alternatives. Try increasing the allowed imbalance, or use a more lenient model?";
  }

  std::map<std::string, unsigned> histogram;
};

Alternatives generate_alternatives(
  const Problem&,
  const Model&,
  unsigned alternatives_count,
  unsigned random_seed,
  std::optional<float> max_imbalance = std::nullopt
);

} // namespace lincs

#endif  // LINCS__GENERATION_HPP
