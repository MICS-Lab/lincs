#include <map>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <lov-e.hpp>

#include "randomness-utils.hpp"


namespace lincs {

struct Domain {
  struct Criterion {
    std::string name;

    enum class ValueType {
      real,
      // @todo Add integer
      // @todo Add enumerated
    } value_type;

    enum class CategoryCorrelation {
      growing,
      // @todo Add decreasing
      // @todo Add single-peaked
      // @todo Add single-valleyed
      // @todo Add unknown
    } category_correlation;

    // @todo Remove these constructors
    // The struct is usable without them in C++, and they were added only to allow using bp::init in the Python module
    // (Do it for other structs as well)
    Criterion() {}
    Criterion(const std::string& name_, ValueType value_type_, CategoryCorrelation category_correlation_): name(name_), value_type(value_type_), category_correlation(category_correlation_) {}

    // @todo Remove this operator
    // The struct is usable without it in C++, and it was added only to allow using bp::vector_indexing_suite in the Python module
    // (Do it for other structs as well)
    bool operator==(const Criterion& other) const {
      return name == other.name && value_type == other.value_type && category_correlation == other.category_correlation;
    }
  };

  std::vector<Criterion> criteria;

  struct Category {
    std::string name;

    Category() {}
    Category(const std::string& name_): name(name_) {}

    bool operator==(const Category& other) const { return name == other.name; }
  };

  std::vector<Category> categories;

  Domain(const std::vector<Criterion>& criteria_, const std::vector<Category>& categories_): criteria(criteria_), categories(categories_) {}

  void dump(std::ostream&) const;
  static Domain load(std::istream&);

  static Domain generate(unsigned criteria_count, unsigned categories_count, unsigned random_seed);
};

struct Model {
  const Domain& domain;

  struct SufficientCoalitions {
    // Sufficient coalitions form an https://en.wikipedia.org/wiki/Upper_set in the set of parts of the set of criteria.
    // This upset can be defined:
    enum class Kind {
      weights,  // by the weights of the criteria
      // @todo Add upset_roots,  // explicitly by its roots
    } kind;

    std::vector<float> criterion_weights;

    SufficientCoalitions() {};
    SufficientCoalitions(Kind kind_, const std::vector<float>& criterion_weights_): kind(kind_), criterion_weights(criterion_weights_) {}
  };

  struct Boundary {
    std::vector<float> profile;
    SufficientCoalitions sufficient_coalitions;

    Boundary() {};
    Boundary(const std::vector<float>& profile_, const SufficientCoalitions& sufficient_coalitions_): profile(profile_), sufficient_coalitions(sufficient_coalitions_) {}

    bool operator==(const Boundary& other) const { return profile == other.profile && sufficient_coalitions.kind == other.sufficient_coalitions.kind && sufficient_coalitions.criterion_weights == other.sufficient_coalitions.criterion_weights; }
  };

  std::vector<Boundary> boundaries;  // boundary_index 0 is between category_index 0 and category_index 1

  Model(const Domain& domain_, const std::vector<Boundary>& boundaries_) : domain(domain_), boundaries(boundaries_) {}

  void dump(std::ostream&) const;
  static Model load(const Domain&, std::istream&);

  static Model generate_mrsort(const Domain&, unsigned random_seed, std::optional<float> fixed_weights_sum = std::nullopt);
};

struct Alternative {
  std::string name;
  std::vector<float> profile;
  std::optional<std::string> category;

  Alternative() {}
  Alternative(const std::string& name_, const std::vector<float>& profile_, const std::optional<std::string>& category_): name(name_), profile(profile_), category(category_) {}

  bool operator==(const Alternative& other) const { return name == other.name && profile == other.profile && category == other.category; }
};

class BalancedAlternativesGenerationException : public std::exception {
 public:
  explicit BalancedAlternativesGenerationException(const std::map<std::string, unsigned>& histogram_) : histogram(histogram_) {}

  const char* what() const noexcept override {
    return "Unable to generate balanced alternatives. Try increasing the allowed imbalance, or use a more lenient model?";
  }

  std::map<std::string, unsigned> histogram;
};

struct Alternatives {
  const Domain& domain;
  std::vector<Alternative> alternatives;

  Alternatives(const Domain& domain_, const std::vector<Alternative>& alternatives_): domain(domain_), alternatives(alternatives_) {}

  void dump(std::ostream&) const;
  static Alternatives load(const Domain&, std::istream&);

  static Alternatives generate(
    const Domain&,
    const Model&,
    unsigned alternatives_count,
    unsigned random_seed,
    std::optional<float> max_imbalance = std::nullopt
  );
};

struct ClassificationResult {
  unsigned unchanged;
  unsigned changed;
};

ClassificationResult classify_alternatives(const Domain&, const Model&, Alternatives*);

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
