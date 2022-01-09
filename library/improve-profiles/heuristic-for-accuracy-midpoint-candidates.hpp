// Copyright 2021-2022 Vincent Jacques

#ifndef IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_MIDPOINT_CANDIDATES_HPP_
#define IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_MIDPOINT_CANDIDATES_HPP_

#include <memory>

#include <chrones.hpp>

#include "../improve-profiles.hpp"
#include "../randomness.hpp"


namespace ppl {

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
with pre-computed candidates
*/
class ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu(
      RandomNumberGenerator random,
      std::shared_ptr<Candidates<Host>> candidates) :
    _random(random),
    _candidates(candidates) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  RandomNumberGenerator _random;
  std::shared_ptr<Candidates<Host>> _candidates;
};

/*
Implement 3.3.4 (variant 2) of https://tel.archives-ouvertes.fr/tel-01370555/document
on the GPU
with pre-computed candidates
*/
class ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu : public ProfilesImprovementStrategy {
 public:
  ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu(
      RandomNumberGenerator random,
      std::shared_ptr<Models<Device>> device_models,
      std::shared_ptr<Candidates<Device>> device_candidates) :
    _random(random),
    _device_models(device_models),
    _device_candidates(device_candidates) {}

  void improve_profiles(std::shared_ptr<Models<Host>>) override;

 private:
  RandomNumberGenerator _random;
  std::shared_ptr<Models<Device>> _device_models;
  std::shared_ptr<Candidates<Device>> _device_candidates;
};

}  // namespace ppl

#endif  // IMPROVE_PROFILES_HEURISTIC_FOR_ACCURACY_MIDPOINT_CANDIDATES_HPP_
