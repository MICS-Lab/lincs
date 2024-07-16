// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__PRE_PROCESSING_HPP
#define LINCS__LEARNING__PRE_PROCESSING_HPP

#include <map>
#include <vector>

#include "../vendored/lov-e.hpp"
#include "../io.hpp"


namespace lincs {

struct PreprocessedBoundary {
  std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>> profile_ranks;
  SufficientCoalitions sufficient_coalitions;

  PreprocessedBoundary(const std::vector<std::variant<unsigned, std::pair<unsigned, unsigned>>>& profile_ranks_, const SufficientCoalitions& sufficient_coalitions_) :
    profile_ranks(profile_ranks_),
    sufficient_coalitions(sufficient_coalitions_)
  {}
};

class PreprocessedLearningSet {
 public:
  // Not copyable
  PreprocessedLearningSet(const PreprocessedLearningSet&) = delete;
  PreprocessedLearningSet& operator=(const PreprocessedLearningSet&) = delete;

  // Movable
  PreprocessedLearningSet(PreprocessedLearningSet&&) = default;
  PreprocessedLearningSet& operator=(PreprocessedLearningSet&&) = default;

  PreprocessedLearningSet(const Problem&, const Alternatives&);

 public:
  Model post_process(const std::vector<PreprocessedBoundary>&) const;

 private:
  const Problem& problem;
 public:
  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned boundaries_count;
  const unsigned alternatives_count;
 private:
  std::map<unsigned, std::vector<float>> real_sorted_values;  // Indexed by [criterion_index][value_rank]
  std::map<unsigned, std::vector<unsigned>> integer_sorted_values;  // Indexed by [criterion_index][value_rank]
 public:
  Array1D<Host, bool> single_peaked;  // [criterion_index]
  Array1D<Host, unsigned> values_counts;  // [criterion_index]
  Array2D<Host, unsigned> performance_ranks;  // [criterion_index][alternative_index]
  Array1D<Host, unsigned> assignments;  // [alternative_index]
};

}  // namespace lincs

#endif  // LINCS__LEARNING__PRE_PROCESSING_HPP
