// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_
#define IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_

#include "../improve-profiles.hpp"


namespace ppl {

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ImproveProfilesWithAccuracyHeuristicOnCpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicOnCpu(
      RandomNumberGenerator random,
      Models<Host>* models) :
#ifndef NDEBUG
    _models(models),
#endif
    _random(random) {}

  void improve_profiles(Models<Host>* models) override {
    CHRONE();

    assert(models == _models);

    _profiles_improver.improve_profiles(_random, models);
  };

 private:
#ifndef NDEBUG
  const Models<Host>* const _models;
#endif
  RandomNumberGenerator _random;
  ProfilesImprover _profiles_improver;
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
on the GPU
*/
class ImproveProfilesWithAccuracyHeuristicOnGpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicOnGpu(
      RandomNumberGenerator random,
      Models<Host>* host_models,
      Models<Device>* device_models) :
#ifndef NDEBUG
    _host_models(host_models),
#endif
    _random(random),
    _device_models(device_models) {}

  void improve_profiles(Models<Host>* host_models) override {
    CHRONE();

    assert(host_models == _host_models);

    replicate_models(*host_models, _device_models);
    _profiles_improver.improve_profiles(_random, _device_models);
    replicate_profiles(*_device_models, host_models);
  };

 private:
#ifndef NDEBUG
  const Models<Host>* const _host_models;
#endif
  RandomNumberGenerator _random;
  Models<Device>* _device_models;
  ProfilesImprover _profiles_improver;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_HPP_
