// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_CPU_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_CPU_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"
#include "accuracy-heuristic/desirability.hpp"


namespace lincs {

class ImproveProfilesWithAccuracyHeuristicOnCpu : public LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnCpu(LearningData& learning_data_) : learning_data(learning_data_) {}

 public:
  void improve_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) override;

 private:
  void improve_model_profiles(unsigned model_index);

  void improve_boundary_profiles(
    unsigned model_index,
    unsigned profile_index,
    ArrayView1D<Host, const unsigned> criterion_indexes
  );

  void improve_low_profile_then_high_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index
  );

  void improve_low_profile_only(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index
  );

  void improve_low_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned lowest_destination_rank,
    unsigned highest_destination_rank
  );

  Desirability compute_move_desirability_for_low_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned destination_rank
  );

  void update_move_desirability_for_low_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned destination_rank,
    unsigned alternative_index,
    Desirability* desirability
  );

  void improve_high_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned lowest_destination_rank,
    unsigned highest_destination_rank
  );

  Desirability compute_move_desirability_for_high_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned destination_rank
  );

  void update_move_desirability_for_high_profile(
    unsigned model_index,
    unsigned profile_index,
    unsigned criterion_index,
    unsigned destination_rank,
    unsigned alternative_index,
    Desirability* desirability
  );

 private:
  LearningData& learning_data;

  static const unsigned max_destinations_count = 64;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_CPU_HPP
