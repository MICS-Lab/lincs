// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_RANDOM_CANDIDATES_HPP_
#define IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_RANDOM_CANDIDATES_HPP_

#include <memory>

#include <chrones.hpp>

#include "../improve-profiles.hpp"
#include "../randomness.hpp"


namespace ppl {

struct Desirability {
  uint v = 0;
  uint w = 0;
  uint q = 0;
  uint r = 0;
  uint t = 0;

  __host__ __device__
  float value() const {
    if (v + w + t + q + r == 0) {
      // The move has no impact. @todo What should its desirability be?
      return 0;
    } else {
      return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
    }
  }
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
*/
class ImproveProfilesWithAccuracyHeuristicOnCpu : public ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristicOnCpu(RandomNumberGenerator random) : _random(random) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  RandomNumberGenerator _random;
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
on the GPU
*/
class ImproveProfilesWithAccuracyHeuristicOnGpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicOnGpu(
      RandomNumberGenerator random,
      std::shared_ptr<Models<Device>> device_models) :
    _random(random),
    _device_models(device_models) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  RandomNumberGenerator _random;
  std::shared_ptr<Models<Device>> _device_models;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_RANDOM_CANDIDATES_HPP_
