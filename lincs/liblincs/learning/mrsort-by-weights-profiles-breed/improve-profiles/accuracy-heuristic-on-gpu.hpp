// Copyright 2023-2024 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP

#include "../../mrsort-by-weights-profiles-breed.hpp"
#include "accuracy-heuristic/desirability.hpp"


namespace lincs {

#ifdef LINCS_HAS_NVCC

class ImproveProfilesWithAccuracyHeuristicOnGpu : public LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy {
 private:
  struct GpuLearningData {
    GpuLearningData(const LearningData&);

    Array2D<Device, unsigned> performance_ranks;  // Indexed by [criterion_index][alternative_index]
    Array1D<Device, unsigned> assignments;  // [alternative_index]
    Array1D<Device, bool> single_peaked;  // [criterion_index]
    Array2D<Device, float> weights;  // [criterion_index][model_index]
    Array3D<Device, unsigned> low_profile_ranks;  // [criterion_index][boundary_index][model_index]
    Array3D<Device, unsigned> high_profile_ranks;  // [criterion_index][boundary_index][model_index]

    Array2D<Device, Desirability> desirabilities;  // [model_index][desination_index]
    Array2D<Device, unsigned> destination_ranks;  // [model_index][desination_index]
  };

 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnGpu(LearningData& host_learning_data_) : host_learning_data(host_learning_data_), gpu_learning_data(host_learning_data) {}

 public:
  void improve_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) override;

 private:
  void improve_model_profiles(unsigned model_index);

  void improve_boundary_profiles(
    unsigned model_index,
    unsigned boundary_index,
    ArrayView1D<Host, const unsigned> criterion_indexes
  );

  void improve_low_profile_then_high_profile(
    unsigned model_index,
    unsigned boundary_index,
    unsigned criterion_index
  );

  void improve_low_profile_only(
    unsigned model_index,
    unsigned boundary_index,
    unsigned criterion_index
  );

  void improve_low_profile(
    unsigned model_index,
    unsigned boundary_index,
    unsigned criterion_index,
    unsigned lowest_destination_rank,
    unsigned highest_destination_rank
  );

  void improve_high_profile(
    unsigned model_index,
    unsigned boundary_index,
    unsigned criterion_index,
    unsigned lowest_destination_rank,
    unsigned highest_destination_rank
  );

 private:
  LearningData& host_learning_data;
  GpuLearningData gpu_learning_data;

  static const unsigned max_destinations_count = 64;
};

#endif  // LINCS_HAS_NVCC

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP
