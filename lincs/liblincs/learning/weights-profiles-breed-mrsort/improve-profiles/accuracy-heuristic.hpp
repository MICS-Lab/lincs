// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_HPP

#include "../../weights-profiles-breed-mrsort.hpp"


namespace lincs {

class ImproveProfilesWithAccuracyHeuristic : public WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristic(Models& models_) : models(models_) {}

 public:
  void improve_profiles() override;

 private:
  struct Desirability {
    // Value for moves with no impact.
    // @todo Verify with Vincent Mousseau that this is the correct value.
    static constexpr float zero_value = 0;

    unsigned v = 0;
    unsigned w = 0;
    unsigned q = 0;
    unsigned r = 0;
    unsigned t = 0;

    float value() const;
  };

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
    const Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const float destination
  );

  void update_move_desirability(
    const Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const float destination,
    const unsigned alternative_index,
    Desirability* desirability
  );

  template<typename T>
  void shuffle(const unsigned model_index, ArrayView1D<Host, T> m) {
    for (unsigned i = 0; i != m.s0(); ++i) {
      std::swap(m[i], m[std::uniform_int_distribution<unsigned int>(0, m.s0() - 1)(models.urbgs[model_index])]);
    }
  }

 private:
  Models& models;
};

}  // namespace lincs

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC_HPP
