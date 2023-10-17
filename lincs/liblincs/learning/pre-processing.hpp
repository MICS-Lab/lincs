// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__PRE_PROCESSING_HPP
#define LINCS__LEARNING__PRE_PROCESSING_HPP

#include <map>
#include <vector>

#include "../vendored/lov-e.hpp"
#include "../io.hpp"


namespace lincs {

struct PreProcessedLearningSet {
  // Not copyable
  PreProcessedLearningSet(const PreProcessedLearningSet&) = delete;
  PreProcessedLearningSet& operator=(const PreProcessedLearningSet&) = delete;

  // Movable
  PreProcessedLearningSet(PreProcessedLearningSet&&) = default;
  PreProcessedLearningSet& operator=(PreProcessedLearningSet&&) = default;

  PreProcessedLearningSet(const Problem&, const Alternatives&);

  const Problem& problem;
  const Alternatives& learning_set;
  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned boundaries_count;
  const unsigned alternatives_count;
  Array2D<Host, float> sorted_values;  // Indexed by [criterion_index][value_rank]
  Array1D<Host, unsigned> values_counts;  // [criterion_index]
  Array2D<Host, unsigned> performance_ranks;  // [criterion_index][alternative_index]
  Array1D<Host, unsigned> assignments;  // [alternative_index]
};

}  // namespace lincs

#endif  // LINCS__LEARNING__PRE_PROCESSING_HPP
