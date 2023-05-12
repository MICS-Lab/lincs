#ifndef LINCS_LINCS_HPP
#define LINCS_LINCS_HPP

#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <lov-e.hpp>

#include "io.hpp"
#include "generation.hpp"
#include "randomness-utils.hpp"
#include "classification.hpp"


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
  const Domain& domain;
  unsigned categories_count;
  unsigned criteria_count;
  unsigned learning_alternatives_count;
  Array2D<Host, float> learning_alternatives;
  Array1D<Host, unsigned> learning_assignments;
  unsigned models_count;
  Array2D<Host, float> weights;
  Array3D<Host, float> profiles;
  std::vector<std::mt19937> urbgs;

  static Models make(const Domain& domain, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed);

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

class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public WeightsProfilesBreedMrSortLearning::ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(Models& models_);

 public:
  void initialize_profiles(
    std::vector<unsigned>::const_iterator model_indexes_begin,
    const std::vector<unsigned>::const_iterator model_indexes_end
  ) override;

 private:
  std::map<float, double> get_candidate_probabilities(
    unsigned criterion_index,
    unsigned profile_index
  );

 private:
  Models& models;
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> generators;
};

class OptimizeWeightsUsingGlop : public WeightsProfilesBreedMrSortLearning::WeightsOptimizationStrategy {
 public:
  OptimizeWeightsUsingGlop(Models& models_) : models(models_) {}

 public:
  void optimize_weights() override;

 private:
  void optimize_model_weights(unsigned model_index);

  struct LinearProgram;

  std::shared_ptr<LinearProgram> make_internal_linear_program(const float epsilon, unsigned model_index);

  auto solve_linear_program(std::shared_ptr<LinearProgram> lp);

 private:
  Models& models;
};

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

class TerminateAtAccuracy : public WeightsProfilesBreedMrSortLearning::TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(unsigned target_accuracy) : _target_accuracy(target_accuracy) {}

 public:
  bool terminate(unsigned, unsigned) override;

 private:
  unsigned _target_accuracy;
};

}  // namespace lincs

#endif  // LINCS_LINCS_HPP
