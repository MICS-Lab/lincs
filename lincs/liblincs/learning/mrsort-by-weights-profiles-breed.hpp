// Copyright 2023-2024 Vincent Jacques

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

  struct ModelsBeingLearned;
  struct ProfilesInitializationStrategy;
  struct WeightsOptimizationStrategy;
  struct ProfilesImprovementStrategy;
  struct BreedingStrategy;
  struct TerminationStrategy;
  struct Observer;

 public:
  LearnMrsortByWeightsProfilesBreed(
    const PreprocessedLearningSet& preprocessed_learning_set_,
    ModelsBeingLearned& models_being_learned_,
    ProfilesInitializationStrategy& profiles_initialization_strategy_,
    WeightsOptimizationStrategy& weights_optimization_strategy_,
    ProfilesImprovementStrategy& profiles_improvement_strategy_,
    BreedingStrategy& breeding_strategy_,
    TerminationStrategy& termination_strategy_,
    const std::vector<Observer*>& observers_ = {}
  ) :
    preprocessed_learning_set(preprocessed_learning_set_),
    models_being_learned(models_being_learned_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    weights_optimization_strategy(weights_optimization_strategy_),
    profiles_improvement_strategy(profiles_improvement_strategy_),
    breeding_strategy(breeding_strategy_),
    termination_strategy(termination_strategy_),
    observers(observers_)
  {}

 public:
  Model perform();

 private:
  unsigned compute_accuracy(unsigned model_index);
  bool is_correctly_assigned(unsigned model_index, unsigned alternative_index);

 public:
  static bool is_accepted(const PreprocessedLearningSet& preprocessed_learning_set, const ModelsBeingLearned&, unsigned model_index, unsigned boundary_index, unsigned criterion_index, unsigned alternative_index);
  static unsigned get_assignment(const PreprocessedLearningSet& preprocessed_learning_set, const ModelsBeingLearned&, unsigned model_index, unsigned alternative_index);

 private:
  const PreprocessedLearningSet& preprocessed_learning_set;
  ModelsBeingLearned& models_being_learned;
  ProfilesInitializationStrategy& profiles_initialization_strategy;
  WeightsOptimizationStrategy& weights_optimization_strategy;
  ProfilesImprovementStrategy& profiles_improvement_strategy;
  BreedingStrategy& breeding_strategy;
  TerminationStrategy& termination_strategy;
  std::vector<Observer*> observers;
};

struct LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned {
  const PreprocessedLearningSet& preprocessed_learning_set;
  const unsigned models_count;
  std::vector<std::mt19937> random_generators;  // [model_index]
  unsigned iteration_index;
  std::vector<unsigned> model_indexes;  // [model_index_index]: this is a reordering of the models' indexes
  Array1D<Host, unsigned> accuracies;  // [model_index]
  Array3D<Host, unsigned> low_profile_ranks;  // [model_index][boundary_index][criterion_index]
  const unsigned single_peaked_criteria_count;
  Array1D<Host, unsigned> high_profile_rank_indexes;  // [criterion_index], meaningful only for single-peaked criteria (i.e. when single_peaked[criterion_index] is true)
  Array3D<Host, unsigned> high_profile_ranks;  // [model_index][boundary_index][high_profile_rank_indexes[criterion_index]]
  Array2D<Host, float> weights;  // [model_index][criterion_index]
  // @todo(Performance, later) Add models' ages

  ModelsBeingLearned(const PreprocessedLearningSet& preprocessed_learning_set, unsigned models_count, unsigned random_seed);

  unsigned get_best_accuracy() const { return accuracies[model_indexes.back()]; }
  Model get_best_model() const { return get_model(model_indexes.back()); }

  Model get_model(unsigned model_index) const;

  #ifndef NDEBUG
  bool model_is_correct(unsigned model_index) const;
  bool models_are_correct() const;
  #endif

 private:
  unsigned count_single_peaked_criteria() const;
};

struct LearnMrsortByWeightsProfilesBreed::ProfilesInitializationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  ProfilesInitializationStrategy(bool supports_single_peaked_criteria_ = false) : supports_single_peaked_criteria(supports_single_peaked_criteria_) {}

  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) = 0;

  const bool supports_single_peaked_criteria;
};

struct LearnMrsortByWeightsProfilesBreed::WeightsOptimizationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  WeightsOptimizationStrategy(bool supports_single_peaked_criteria_ = false) : supports_single_peaked_criteria(supports_single_peaked_criteria_) {}

  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights(unsigned model_indexes_begin, unsigned model_indexes_end) = 0;

  const bool supports_single_peaked_criteria;
};

struct LearnMrsortByWeightsProfilesBreed::ProfilesImprovementStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  ProfilesImprovementStrategy(bool supports_single_peaked_criteria_ = false) : supports_single_peaked_criteria(supports_single_peaked_criteria_) {}

  virtual ~ProfilesImprovementStrategy() {}

  virtual void improve_profiles(unsigned model_indexes_begin, unsigned model_indexes_end) = 0;

  const bool supports_single_peaked_criteria;
};

struct LearnMrsortByWeightsProfilesBreed::BreedingStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  BreedingStrategy(bool supports_single_peaked_criteria_ = false) : supports_single_peaked_criteria(supports_single_peaked_criteria_) {}

  virtual ~BreedingStrategy() {}

  virtual void breed() = 0;

  const bool supports_single_peaked_criteria;
};

struct LearnMrsortByWeightsProfilesBreed::TerminationStrategy {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  virtual ~TerminationStrategy() {}

  virtual bool terminate() = 0;
};

struct LearnMrsortByWeightsProfilesBreed::Observer {
  typedef LearnMrsortByWeightsProfilesBreed::ModelsBeingLearned ModelsBeingLearned;

  virtual ~Observer() {}

  virtual void after_iteration() = 0;
  virtual void before_return() = 0;
};

}  // namespace lincs


#endif  // LINCS__LEARNING__MRSORT_BY_WEIGHTS_PROFILES_BREED_HPP
