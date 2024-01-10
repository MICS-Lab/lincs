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
  void improve_profiles() override;

 private:
  void improve_model_profiles(const unsigned model_index);

  void improve_model_profile(
    const unsigned model_index,
    const unsigned profile_index,
    ArrayView1D<Host, const unsigned> criterion_indexes
  );

  void improve_model_profile(
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index
  );

  Desirability compute_move_desirability(
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const unsigned destination_rank
  );

  void update_move_desirability(
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const unsigned destination_rank,
    const unsigned alternative_index,
    Desirability* desirability
  );

 private:
  LearningData& learning_data;

  static const unsigned max_destinations_count = 64;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_CPU_HPP
