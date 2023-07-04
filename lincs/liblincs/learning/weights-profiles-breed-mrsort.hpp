// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP

#include <random>

#include "../io.hpp"
#include "../vendored/lov-e.hpp"


namespace lincs {

class WeightsProfilesBreedMrSortLearning {
 public:
  static const unsigned default_models_count = 9;

  struct LearningData;
  struct ProfilesInitializationStrategy;
  struct WeightsOptimizationStrategy;
  struct ProfilesImprovementStrategy;
  struct BreedingStrategy;
  struct TerminationStrategy;

 public:
  WeightsProfilesBreedMrSortLearning(
    LearningData& learning_data_,
    ProfilesInitializationStrategy& profiles_initialization_strategy_,
    WeightsOptimizationStrategy& weights_optimization_strategy_,
    ProfilesImprovementStrategy& profiles_improvement_strategy_,
    BreedingStrategy& breeding_strategy_,
    TerminationStrategy& termination_strategy_
  ) :
    learning_data(learning_data_),
    profiles_initialization_strategy(profiles_initialization_strategy_),
    weights_optimization_strategy(weights_optimization_strategy_),
    profiles_improvement_strategy(profiles_improvement_strategy_),
    breeding_strategy(breeding_strategy_),
    termination_strategy(termination_strategy_) {}

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
};

struct WeightsProfilesBreedMrSortLearning::LearningData {
  const Problem& problem;
  unsigned categories_count;
  unsigned criteria_count;
  unsigned learning_alternatives_count;
  Array2D<Host, float> learning_alternatives;
  Array1D<Host, unsigned> learning_assignments;
  unsigned iteration_index;
  unsigned models_count;
  std::vector<unsigned> model_indexes;
  Array2D<Host, float> weights;
  Array3D<Host, float> profiles;
  Array1D<Host, unsigned> accuracies;
  std::vector<std::mt19937> urbgs;

  static LearningData make(const Problem& problem, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed);

  unsigned get_best_accuracy() const { return accuracies[model_indexes.back()]; }

  Model get_model(const unsigned model_index) const;
};

struct WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::LearningData LearningData;

  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    std::vector<unsigned>::const_iterator model_indexes_begin,
    std::vector<unsigned>::const_iterator model_indexes_end) = 0;
};

struct WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::LearningData LearningData;

  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights() = 0;
};

struct WeightsProfilesBreedMrSortLearning::ProfilesImprovementStrategy {
  typedef WeightsProfilesBreedMrSortLearning::LearningData LearningData;

  virtual ~ProfilesImprovementStrategy() {}

  virtual void improve_profiles() = 0;
};

struct WeightsProfilesBreedMrSortLearning::BreedingStrategy {
  typedef WeightsProfilesBreedMrSortLearning::LearningData LearningData;

  virtual ~BreedingStrategy() {}

  virtual void breed() = 0;
};

struct WeightsProfilesBreedMrSortLearning::TerminationStrategy {
  typedef WeightsProfilesBreedMrSortLearning::LearningData LearningData;

  virtual ~TerminationStrategy() {}

  virtual bool terminate() = 0;
};

}  // namespace lincs


#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT_HPP
