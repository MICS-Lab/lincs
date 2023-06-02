// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class ImproveProfilesWithAccuracyHeuristicOnGpu : public WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy {
 public:
  struct GpuModels;

 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnGpu(Models& host_models_, GpuModels& gpu_models_) : host_models(host_models_), gpu_models(gpu_models_) {}

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

  unsigned get_assignment(
    const unsigned model_index,
    const unsigned alternative_index
  );

  template<typename T>
  void shuffle(const unsigned model_index, ArrayView1D<Host, T> m) {
    for (unsigned i = 0; i != m.s0(); ++i) {
      std::swap(m[i], m[std::uniform_int_distribution<unsigned int>(0, m.s0() - 1)(host_models.urbgs[model_index])]);
    }
  }

 private:
  Models& host_models;
  GpuModels& gpu_models;  // @todo Rename device_models?
};

struct ImproveProfilesWithAccuracyHeuristicOnGpu::GpuModels {
  unsigned categories_count;
  unsigned criteria_count;
  unsigned learning_alternatives_count;
  Array2D<Device, float> learning_alternatives;
  Array1D<Device, unsigned> learning_assignments;
  unsigned models_count;
  Array2D<Device, float> weights;
  Array3D<Device, float> profiles;

  static GpuModels make(const Models&);
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_ON_GPU_HPP
