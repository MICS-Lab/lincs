// Copyright 2021 Vincent Jacques

#ifndef IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_
#define IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_

#include "../improve-profiles.hpp"


namespace ppl {

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ImproveProfilesWithAccuracyHeuristicOnCpu : public ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnCpu(Models<Host>* models) : _models(models) {}

  void improve_profiles(const RandomSource& random /* @todo Put in ctor */) override {
    _profiles_improver.improve_profiles(random, _models);
  };

 private:
  Models<Host>* _models;
  ProfilesImprover _profiles_improver;
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
on the GPU
*/
class ImproveProfilesWithAccuracyHeuristicOnGpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicOnGpu(
      Models<Host>* host_models,
      Models<Device>* device_models) :
    _host_models(host_models),
    _device_models(device_models) {}

  void improve_profiles(const RandomSource& random /* @todo Put in ctor */) override {
    replicate_models(*_host_models, _device_models);
    _profiles_improver.improve_profiles(random, _device_models);
    replicate_profiles(*_device_models, _host_models);
  };

 private:
  Models<Host>* _host_models;
  Models<Device>* _device_models;
  ProfilesImprover _profiles_improver;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_
