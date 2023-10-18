// Copyright 2023 Vincent Jacques

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
    Array2D<Device, float> weights;  // [criterion_index][model_index]
    Array3D<Device, unsigned> profile_ranks;  // [criterion_index][profile_index][model_index]

    Array2D<Device, Desirability> desirabilities;  // [model_index][desination_index]
    Array2D<Device, unsigned> destination_ranks;  // [model_index][desination_index]
  };

 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnGpu(LearningData& host_learning_data_) : host_learning_data(host_learning_data_), gpu_learning_data(host_learning_data) {}

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

 private:
  LearningData& host_learning_data;
  GpuLearningData gpu_learning_data;

  static const unsigned max_destinations_count = 64;
};

#endif  // LINCS_HAS_NVCC

}  // namespace lincs

#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP
