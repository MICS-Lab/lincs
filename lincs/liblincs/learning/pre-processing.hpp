// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__PRE_PROCESSING_HPP
#define LINCS__LEARNING__PRE_PROCESSING_HPP

#include <map>
#include <vector>

#include "../vendored/lov-e.hpp"
#include "../io.hpp"


namespace lincs {

struct PreProcessedModel {
  struct Boundary {
    std::vector<unsigned> profile_ranks;
    SufficientCoalitions sufficient_coalitions;

    Boundary(const std::vector<unsigned>& profile_ranks_, const SufficientCoalitions& sufficient_coalitions_) :
      profile_ranks(profile_ranks_),
      sufficient_coalitions(sufficient_coalitions_)
    {}
  };

  std::vector<Boundary> boundaries;
};

class PreProcessedLearningSet {
 public:
  // Not copyable
  PreProcessedLearningSet(const PreProcessedLearningSet&) = delete;
  PreProcessedLearningSet& operator=(const PreProcessedLearningSet&) = delete;

  // Movable
  PreProcessedLearningSet(PreProcessedLearningSet&&) = default;
  PreProcessedLearningSet& operator=(PreProcessedLearningSet&&) = default;

  PreProcessedLearningSet(const Problem&, const Alternatives&);

 public:
  // @todo(Project management, soon) Remove 'do_halves'; homogenize behavior
  Model post_process(const PreProcessedModel&, bool do_halves=true) const;

 private:
  const Problem& problem;
 public:
  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned boundaries_count;
  const unsigned alternatives_count;
 private:
  Array2D<Host, float> sorted_values;  // Indexed by [criterion_index][value_rank]
 public:
  Array1D<Host, unsigned> values_counts;  // [criterion_index]
  Array2D<Host, unsigned> performance_ranks;  // [criterion_index][alternative_index]
  Array1D<Host, unsigned> assignments;  // [alternative_index]
};

}  // namespace lincs

#endif  // LINCS__LEARNING__PRE_PROCESSING_HPP
