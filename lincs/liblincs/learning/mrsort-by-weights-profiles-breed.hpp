// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED_HPP
#define LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED_HPP

#include <random>

#include "../io.hpp"
#include "pre-processing.hpp"
#include "../vendored/lov-e.hpp"


namespace lincs {

class LearnMrsortByWeightsProfilesBreed {
 public:
  static const unsigned default_models_count = 9;

  struct LearningData;
  struct ProfilesInitializationStrategy;
  struct WeightsOptimizationStrategy;
  struct ProfilesImprovementStrategy;
  struct BreedingStrategy;
  struct TerminationStrategy;
  struct Observer;

 public:
  LearnMrsortByWeightsProfilesBreed(
    LearningData& learning_data_,
    ProfilesInitializationStrategy& profiles_initialization_strategy_,
    WeightsOptimizationStrategy& weights_optimization_strategy_,
    ProfilesImprovementStrategy& profiles_improvement_strategy_,
    BreedingStrategy& breeding_strategy_,
    TerminationStrategy& termination_strategy_,
    const std::vector<Observer*>& observers_ = {}
  ) :
    learning_data(learning_data_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    weights_optimization_strategy(weights_optimization_strategy_),
    profiles_improvement_strategy(profiles_improvement_strategy_),
    breeding_strategy(breeding_strategy_),
    termination_strategy(termination_strategy_),
    observers(observers_) {}

 public:
  Model perform();

 private:
  unsigned compute_accuracy(const unsigned model_index);
  bool is_correctly_assigned(const unsigned model_index, const unsigned alternative_index);

 public:
  static unsigned get_assignment(const LearningData& learning_data, const unsigned model_index, const unsigned alternative_index);

 private:
  LearningData& learning_data;
  ProfilesInitializationStrategy& profiles_initialization_strategy;
  WeightsOptimizationStrategy& weights_optimization_strategy;
  ProfilesImprovementStrategy& profiles_improvement_strategy;
  BreedingStrategy& breeding_strategy;
  TerminationStrategy& termination_strategy;
  std::vector<Observer*> observers;
};

struct LearnMrsortByWeightsProfilesBreed::LearningData : public PreProcessedLearningSet {
  unsigned iteration_index;
  unsigned models_count;
  std::vector<unsigned> model_indexes;  // [model_index_index]: this is a reordering of the models' indexes
  Array2D<Host, float> weights;  // [model_index][criterion_index]
  Array3D<Host, unsigned> profile_ranks;  // [model_index][profile_index][criterion_index]
  Array1D<Host, unsigned> accuracies;  // [model_index]
  // @todo(Performance, later) Add models' ages
  std::vector<std::mt19937> urbgs;  // [model_index]

  LearningData(const Problem& problem, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed);

  unsigned get_best_accuracy() const { return accuracies[model_indexes.back()]; }

  Model get_model(const unsigned model_index) const;
};

struct LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) = 0;
};

struct LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights() = 0;
};

struct LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~ProfilesImprovementStrategy() {}

  virtual void improve_profiles() = 0;
};

struct LearnMrsortByWeightsProfilesBreed::BreedingStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~BreedingStrategy() {}

  virtual void breed() = 0;
};

struct LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~TerminationStrategy() {}

  virtual bool terminate() = 0;
};

struct LearnMrsortByWeightsProfilesBreed::Observer {
  typedef LearnMrsortByWeightsProfilesBreed::LearningData LearningData;

  virtual ~Observer() {}

  virtual void after_iteration() = 0;
  virtual void before_return() = 0;
};

}  // namespace lincs


#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED_HPP
