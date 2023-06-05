// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP

#include <random>

#include <lov-e.hpp>

#include "../io.hpp"


namespace lincs {

class WeightsProfilesBreedMrSortLearning {
 public:
  static const unsigned default_models_count = 9;

  struct Models;
  struct ProfilesInitializationStrategy;
  struct WeightsOptimizationStrategy;
  struct ProfilesImprovementStrategy;
  struct TerminationStrategy;

 public:
  WeightsProfilesBreedMrSortLearning(
    Models& models_,
    ProfilesInitializationStrategy& profiles_initialization_strategy_,
    WeightsOptimizationStrategy& weights_optimization_strategy_,
    ProfilesImprovementStrategy& profiles_improvement_strategy_,
    TerminationStrategy& termination_strategy_
  ) :
    models(models_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    weights_optimization_strategy(weights_optimization_strategy_),
    profiles_improvement_strategy(profiles_improvement_strategy_),
    termination_strategy(termination_strategy_) {}

 public:
  Model perform();

 private:
  std::pair<std::vector<unsigned>, unsigned> partition_models_by_accuracy();
  unsigned get_accuracy(const unsigned model_index);
  bool is_correctly_assigned(const unsigned model_index, const unsigned alternative_index);

 public:
  static unsigned get_assignment(const Models& models, const unsigned model_index, const unsigned alternative_index);

 private:
  Models& models;
  ProfilesInitializationStrategy& profiles_initialization_strategy;
  WeightsOptimizationStrategy& weights_optimization_strategy;
  ProfilesImprovementStrategy& profiles_improvement_strategy;
  TerminationStrategy& termination_strategy;
};

struct WeightsProfilesBreedMrSortLearning::Models {
  const Problem& problem;
  unsigned categories_count;
  unsigned criteria_count;
  unsigned learning_alternatives_count;
  Array2D<Host, float> learning_alternatives;
  Array1D<Host, unsigned> learning_assignments;
  unsigned models_count;
  Array2D<Host, float> weights;
  Array3D<Host, float> profiles;
  std::vector<std::mt19937> urbgs;

  static Models make(const Problem& problem, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed);

  Model get_model(const unsigned model_index) const;
};

struct WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::Models Models;

  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    std::vector<unsigned>::const_iterator model_indexes_begin,
    std::vector<unsigned>::const_iterator model_indexes_end) = 0;
};

struct WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::Models Models;

  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights() = 0;
};

struct WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy {
  typedef WeightsProfilesBreedMrSortLearning::Models Models;

  virtual ~ProfilesImprovementStrategy() {}

  virtual void improve_profiles() = 0;
};

struct WeightsProfilesBreedMrSortLearning::TerminationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::Models Models;

  virtual ~TerminationStrategy() {}

  virtual bool terminate(unsigned iteration_index, unsigned best_accuracy) = 0;
};

}  // namespace lincs


#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP
