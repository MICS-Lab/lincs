#include "lincs.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <doctest.h>
#include <lov-e.hpp>
#include <magic_enum.hpp>
#include <omp.h>
#include <ortools/glop/lp_solver.h>

#include "median-and-max.hpp"


namespace glp = operations_research::glop;

namespace lincs {

namespace io {

// Classes for easy input/output of domain objects

struct Model {
  Model(
    unsigned criteria_count_,
    unsigned categories_count_,
    const std::vector<std::vector<float>>& profiles_,
    const std::vector<float>& weights_) :
      criteria_count(criteria_count_),
      categories_count(categories_count_),
      profiles(profiles_),
      weights(weights_) {};

  const unsigned criteria_count;
  const unsigned categories_count;
  std::vector<std::vector<float>> profiles;
  std::vector<float> weights;
};

struct ClassifiedAlternative {
  ClassifiedAlternative(
    const std::vector<float>& criteria_values_,
    unsigned assigned_category_) :
      criteria_values(criteria_values_),
      assigned_category(assigned_category_) {}

  const std::vector<float> criteria_values;
  unsigned assigned_category;
};

struct LearningSet {
  LearningSet(
    unsigned criteria_count_,
    unsigned categories_count_,
    unsigned alternatives_count_,
    const std::vector<ClassifiedAlternative>& alternatives_) :
      criteria_count(criteria_count_),
      categories_count(categories_count_),
      alternatives_count(alternatives_count_),
      alternatives(alternatives_) {}

  const unsigned criteria_count;
  const unsigned categories_count;
  const unsigned alternatives_count;
  std::vector<ClassifiedAlternative> alternatives;
};

}  // namespace io

class Random {
 public:
  explicit Random(int seed) : _gen(omp_get_max_threads()) {
    #pragma omp parallel
    {
      urbg().seed(seed * (omp_get_thread_num() + 1));
    }
  }

  // Non-copyable
  Random(const Random&) = delete;
  Random& operator=(const Random&) = delete;
  // Could be made movable if needed
  Random(Random&&) = delete;
  Random& operator=(Random&&) = delete;

  float uniform_float(const float min, const float max) const {
    float v = max;

    do {
      v = std::uniform_real_distribution<float>(min, max)(urbg());
    } while (v == max);

    return v;
  }

  unsigned uniform_int(const unsigned min, const unsigned max) const {
    return std::uniform_int_distribution<unsigned int>(min, max - 1)(urbg());
  }

  std::mt19937& urbg() const {
    const unsigned thread_index = omp_get_thread_num();
    assert(thread_index < _gen.size());
    return _gen[thread_index];
  }

 private:
  mutable std::vector<std::mt19937> _gen;
};

/*
Pick random values from a finite set with given probabilities
(a discrete distribution with arbitrary values).
*/
template<typename T>
class ProbabilityWeightedGenerator {
  ProbabilityWeightedGenerator(const std::vector<T>& values, const std::vector<double>& probabilities) :
    _values(values),
    _distribution(probabilities.begin(), probabilities.end())
  {}

 public:
  static ProbabilityWeightedGenerator make(std::map<T, double> value_probabilities) {
    std::vector<T> values;
    values.reserve(value_probabilities.size());
    std::vector<double> probabilities;
    probabilities.reserve(value_probabilities.size());
    for (auto value_probability : value_probabilities) {
      values.push_back(value_probability.first);
      probabilities.push_back(value_probability.second);
    }
    return ProbabilityWeightedGenerator(values, probabilities);
  }

  std::map<T, double> get_value_probabilities() {
    std::map<T, double> value_probabilities;
    auto probabilities = _distribution.probabilities();
    const unsigned size = _values.size();
    assert(probabilities.size() == size);
    for (unsigned i = 0; i != size; ++i) {
      value_probabilities[_values[i]] = probabilities[i];
    }
    return value_probabilities;
  }

  template<typename Generator>
  T operator()(Generator& gen) const {  // NOLINT(runtime/references)
    const unsigned index = _distribution(gen);
    assert(index < _values.size());
    return _values[index];
  }

 private:
  std::vector<T> _values;
  mutable std::discrete_distribution<unsigned> _distribution;
};

class Domain_ {
 public:
  static std::shared_ptr<Domain_> make(const io::LearningSet& learning_set) {
    Array2D<Host, float> alternatives(learning_set.criteria_count, learning_set.alternatives_count, uninitialized);
    Array1D<Host, unsigned> assignments(learning_set.alternatives_count, uninitialized);

    for (unsigned alt_index = 0; alt_index != learning_set.alternatives_count; ++alt_index) {
      const io::ClassifiedAlternative& alt = learning_set.alternatives[alt_index];

      for (unsigned crit_index = 0; crit_index != learning_set.criteria_count; ++crit_index) {
        alternatives[crit_index][alt_index] = alt.criteria_values[crit_index];
      }

      assignments[alt_index] = alt.assigned_category;
    }

    return std::make_shared<Domain_>(
      learning_set.categories_count,
      learning_set.criteria_count,
      learning_set.alternatives_count,
      std::move(alternatives),
      std::move(assignments));
  }

 public:
  Domain_(
    unsigned categories_count_,
    unsigned criteria_count_,
    unsigned learning_alternatives_count_,
    Array2D<Host, float>&& learning_alternatives_,
    Array1D<Host, unsigned>&& learning_assignments_) :
      categories_count(categories_count_),
      criteria_count(criteria_count_),
      learning_alternatives_count(learning_alternatives_count_),
      learning_alternatives(std::move(learning_alternatives_)),
      learning_assignments(std::move(learning_assignments_)) {}

 public:
  unsigned categories_count;
  unsigned criteria_count;
  unsigned learning_alternatives_count;
  Array2D<Host, float> learning_alternatives;
  Array1D<Host, unsigned> learning_assignments;
};

class Models {
 public:
  static std::shared_ptr<Models> make(Domain_& domain, const unsigned models_count) {
    Array1D<Host, unsigned> initialization_iteration_indexes(models_count, uninitialized);
    Array2D<Host, float> weights(domain.criteria_count, models_count, uninitialized);
    Array3D<Host, float> profiles(
      domain.criteria_count, (domain.categories_count - 1), models_count, uninitialized);

    return std::make_shared<Models>(
      domain,
      models_count,
      std::move(initialization_iteration_indexes),
      std::move(weights),
      std::move(profiles));
  }

  io::Model unmake_one(unsigned model_index) const {
    std::vector<std::vector<float>> model_profiles(domain.categories_count - 1);
    for (unsigned cat_index = 0; cat_index != domain.categories_count - 1; ++cat_index) {
      model_profiles[cat_index].reserve(domain.criteria_count);
      for (unsigned crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
        model_profiles[cat_index].push_back(profiles[crit_index][cat_index][model_index]);
      }
    }

    std::vector<float> model_weights;
    model_weights.reserve(domain.criteria_count);
    for (unsigned crit_index = 0; crit_index != domain.criteria_count; ++crit_index) {
      model_weights.push_back(weights[crit_index][model_index]);
    }

    return io::Model(domain.criteria_count, domain.categories_count, model_profiles, model_weights);
  }

 public:
  Models(
    Domain_& domain_,
    const unsigned models_count_,
    Array1D<Host, unsigned>&& initialization_iteration_indexes_,
    Array2D<Host, float>&& weights_,
    Array3D<Host, float>&& profiles_) :
      domain(domain_),
      models_count(models_count_),
      initialization_iteration_indexes(std::move(initialization_iteration_indexes_)),
      weights(std::move(weights_)),
      profiles(std::move(profiles_)) {}

 public:
  Domain_& domain;
  unsigned models_count;
  Array1D<Host, unsigned> initialization_iteration_indexes;
  Array2D<Host, float> weights;
  Array3D<Host, float> profiles;
};

class ProfilesInitializationStrategy {
 public:
  virtual ~ProfilesInitializationStrategy() {}

  virtual void initialize_profiles(
    std::shared_ptr<Models> models,
    unsigned iteration_index,
    std::vector<unsigned>::const_iterator model_indexes_begin,
    std::vector<unsigned>::const_iterator model_indexes_end) = 0;
};

class ProfilesImprovementStrategy {
 public:
  virtual ~ProfilesImprovementStrategy() {}
  virtual void improve_profiles(std::shared_ptr<Models>) = 0;
};

class WeightsOptimizationStrategy {
 public:
  virtual ~WeightsOptimizationStrategy() {}

  virtual void optimize_weights(std::shared_ptr<Models>) = 0;
};

class TerminationStrategy {
 public:
  virtual ~TerminationStrategy() {}

  virtual bool terminate(unsigned iteration_index, unsigned best_accuracy) = 0;
};

const unsigned default_models_count = 9;

unsigned get_assignment(const Models& models, const unsigned model_index, const unsigned alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index < models.models_count);
  assert(alternative_index < models.domain.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = models.domain.categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alternative_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}

bool is_correctly_assigned(
    const Models& models,
    const unsigned model_index,
    const unsigned alternative_index) {
  const unsigned expected_assignment = models.domain.learning_assignments[alternative_index];
  const unsigned actual_assignment = get_assignment(models, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

unsigned get_accuracy(const Models& models, const unsigned model_index) {
  unsigned accuracy = 0;

  for (unsigned alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
    if (is_correctly_assigned(models, model_index, alt_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

class InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion : public ProfilesInitializationStrategy {
 public:
  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(
    const Random& random,
    const Models& models) :
      _random(random) {
    _generators.reserve(models.domain.categories_count - 1);

    for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      _generators.push_back(std::vector<ProbabilityWeightedGenerator<float>>());
      _generators.back().reserve(models.domain.criteria_count);
      for (unsigned profile_index = 0; profile_index != models.domain.categories_count - 1; ++profile_index) {
        _generators.back().push_back(ProbabilityWeightedGenerator<float>::make(
          get_candidate_probabilities(models.domain, crit_index, profile_index)));
      }
    }
  }

 private:
  std::map<float, double> get_candidate_probabilities(
    const Domain_& domain,
    unsigned crit_index,
    unsigned profile_index
  ) {
    std::vector<float> values_below;
    // The size used for 'reserve' is a few times larger than the actual final size,
    // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
    // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
    values_below.reserve(domain.learning_alternatives_count);
    std::vector<float> values_above;
    values_above.reserve(domain.learning_alternatives_count);
    // This loop could/should be done once outside this function
    for (unsigned alt_index = 0; alt_index != domain.learning_alternatives_count; ++alt_index) {
      const float value = domain.learning_alternatives[crit_index][alt_index];
      const unsigned assignment = domain.learning_assignments[alt_index];
      if (assignment == profile_index) {
        values_below.push_back(value);
      } else if (assignment == profile_index + 1) {
        values_above.push_back(value);
      }
    }

    std::map<float, double> candidate_probabilities;

    for (auto candidates : { values_below, values_above }) {
      for (auto candidate : candidates) {
        if (candidate_probabilities.find(candidate) != candidate_probabilities.end()) {
          // Candidate value has already been evaluated (because it appears several times)
          continue;
        }

        unsigned correctly_classified_count = 0;
        // @todo Could we somehow sort 'values_below' and 'values_above' and walk the values only once?
        // (Transforming this O(n²) loop in O(n*log n) + O(n))
        for (auto value : values_below) if (value < candidate) ++correctly_classified_count;
        for (auto value : values_above) if (value >= candidate) ++correctly_classified_count;
        candidate_probabilities[candidate] = static_cast<double>(correctly_classified_count) / candidates.size();
      }
    }

    return candidate_probabilities;
  }

 public:
  void initialize_profiles(
    std::shared_ptr<Models> models,
    const unsigned iteration_index,
    std::vector<unsigned>::const_iterator model_indexes_begin,
    const std::vector<unsigned>::const_iterator model_indexes_end
  ) override {
    // Embarrassingly parallel
    for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
      const unsigned model_index = *model_indexes_begin;

      models->initialization_iteration_indexes[model_index] = iteration_index;

      // Embarrassingly parallel
      for (unsigned crit_index = 0; crit_index != models->domain.criteria_count; ++crit_index) {
        // Not parallel because of the profiles ordering constraint
        for (unsigned category_index = models->domain.categories_count - 1; category_index != 0; --category_index) {
          const unsigned profile_index = category_index - 1;
          float value = _generators[crit_index][profile_index](_random.urbg());

          if (profile_index != models->domain.categories_count - 2) {
            value = std::min(value, models->profiles[crit_index][profile_index + 1][model_index]);
          }
          // @todo Add a unit test that triggers the following assertion
          // (This will require removing the code to enforce the order of profiles above)
          // Then restore the code to enforce the order of profiles
          // Note, this assertion does not protect us from initializing a model with two identical profiles.
          // Is it really that bad?
          assert(
            profile_index == models->domain.categories_count - 2
            || models->profiles[crit_index][profile_index + 1][model_index] >= value);

          models->profiles[crit_index][profile_index][model_index] = value;
        }
      }
    }
  }

 private:
  const Random& _random;
  std::vector<std::vector<ProbabilityWeightedGenerator<float>>> _generators;
};

template<typename T>
void swap(T& a, T& b) {
  T c = a;
  a = b;
  b = c;
}

template<typename T>
void shuffle(const Random& random, ArrayView1D<Host, T> m) {
  for (unsigned i = 0; i != m.s0(); ++i) {
    swap(m[i], m[random.uniform_int(0, m.s0())]);
  }
}

class ImproveProfilesWithAccuracyHeuristic : public ProfilesImprovementStrategy {
 public:
  explicit ImproveProfilesWithAccuracyHeuristic(const Random& random) : _random(random) {}

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

    float value() const {
      if (v + w + t + q + r == 0) {
        return zero_value;
      } else {
        return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
      }
    }
  };

  void update_move_desirability(
    const Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const float destination,
    const unsigned alt_index,
    Desirability* desirability
  ) {
    const float current_position = models.profiles[criterion_index][profile_index][model_index];
    const float weight = models.weights[criterion_index][model_index];

    const float value = models.domain.learning_alternatives[criterion_index][alt_index];
    const unsigned learning_assignment = models.domain.learning_assignments[alt_index];
    const unsigned model_assignment = get_assignment(models, model_index, alt_index);

    // @todo Factorize with get_assignment
    float weight_at_or_above_profile = 0;
    for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
      const float profile_value = models.profiles[crit_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[crit_index][model_index];
      }
    }

    // These imbricated conditionals could be factorized, but this form has the benefit
    // of being a direct translation of the top of page 78 of Sobrie's thesis.
    // Correspondance:
    // - learning_assignment: bottom index of A*
    // - model_assignment: top index of A*
    // - profile_index: h
    // - destination: b_j +/- \delta
    // - current_position: b_j
    // - value: a_j
    // - weight_at_or_above_profile: \sigma
    // - weight: w_j
    // - 1: \lambda
    if (destination > current_position) {
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight < 1) {
          ++desirability->v;
      }
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight >= 1) {
          ++desirability->w;
      }
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index + 1
        && destination > value
        && value >= current_position
        && weight_at_or_above_profile - weight < 1) {
          ++desirability->q;
      }
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination > value
        && value >= current_position) {
          ++desirability->r;
      }
      if (
        learning_assignment < profile_index
        && model_assignment > profile_index
        && destination > value
        && value >= current_position) {
          ++desirability->t;
      }
    } else {
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight >= 1) {
          ++desirability->v;
      }
      if (
        learning_assignment == profile_index + 1
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight < 1) {
          ++desirability->w;
      }
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index
        && destination < value
        && value < current_position
        && weight_at_or_above_profile + weight >= 1) {
          ++desirability->q;
      }
      if (
        learning_assignment == profile_index
        && model_assignment == profile_index + 1
        && destination <= value
        && value < current_position) {
          ++desirability->r;
      }
      if (
        learning_assignment > profile_index + 1
        && model_assignment < profile_index + 1
        && destination < value
        && value <= current_position) {
          ++desirability->t;
      }
    }
  }

  Desirability compute_move_desirability(
    const Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index,
    const float destination
  ) {
    Desirability d;

    for (unsigned alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
      update_move_desirability(
        models, model_index, profile_index, criterion_index, destination, alt_index, &d);
    }

    return d;
  }

  void improve_model_profile(
    const Random& random,
    Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    const unsigned criterion_index
  ) {
    // WARNING: We're assuming all criteria have values in [0, 1]
    // @todo Can we relax this assumption?
    // This is consistent with our comment in the header file, but slightly less generic than Sobrie's thesis
    const float lowest_destination =
      profile_index == 0 ? 0. :
      models.profiles[criterion_index][profile_index - 1][model_index];
    const float highest_destination =
      profile_index == models.domain.categories_count - 2 ? 1. :
      models.profiles[criterion_index][profile_index + 1][model_index];

    float best_destination = models.profiles[criterion_index][profile_index][model_index];
    float best_desirability = Desirability().value();

    if (lowest_destination == highest_destination) {
      assert(best_destination == lowest_destination);
      return;
    }

    // Not sure about this part: we're considering an arbitrary number of possible moves as described in
    // Mousseau's prez-mics-2018(8).pdf, but:
    //  - this is wasteful when there are fewer alternatives in the interval
    //  - this is not strictly consistent with, albeit much simpler than, Sobrie's thesis
    // @todo Ask Vincent Mousseau about the following:
    // We could consider only a finite set of values for b_j described as follows:
    // - sort all the 'a_j's
    // - compute all midpoints between two successive 'a_j'
    // - add two extreme values (0 and 1, or above the greatest a_j and below the smallest a_j)
    // Then instead of taking a random values in [lowest_destination, highest_destination],
    // we'd take a random subset of the intersection of these midpoints with that interval.
    for (unsigned n = 0; n < 64; ++n) {
      // Map (embarrassingly parallel)
      const float destination = random.uniform_float(lowest_destination, highest_destination);
      const float desirability = compute_move_desirability(
        models, model_index, profile_index, criterion_index, destination).value();
      // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
      if (desirability > best_desirability) {
        best_desirability = desirability;
        best_destination = destination;
      }
    }

    // @todo Desirability can be as high as 2. The [0, 1] interval is a weird choice.
    if (random.uniform_float(0, 1) <= best_desirability) {
      models.profiles[criterion_index][profile_index][model_index] = best_destination;
    }
  }

  void improve_model_profile(
    const Random& random,
    Models& models,
    const unsigned model_index,
    const unsigned profile_index,
    ArrayView1D<Host, const unsigned> criterion_indexes
  ) {
    // Not parallel because iteration N+1 relies on side effect in iteration N
    // (We could challenge this aspect of the algorithm described by Sobrie)
    for (unsigned crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
      improve_model_profile(random, models, model_index, profile_index, criterion_indexes[crit_idx_idx]);
    }
  }

  void improve_model_profiles(const Random& random, Models& models, const unsigned model_index) {
    Array1D<Host, unsigned> criterion_indexes(models.domain.criteria_count, uninitialized);
    // Not worth parallelizing because models.domain.criteria_count is typically small
    for (unsigned crit_idx_idx = 0; crit_idx_idx != models.domain.criteria_count; ++crit_idx_idx) {
      criterion_indexes[crit_idx_idx] = crit_idx_idx;
    }

    // Not parallel because iteration N+1 relies on side effect in iteration N
    // (We could challenge this aspect of the algorithm described by Sobrie)
    for (unsigned profile_index = 0; profile_index != models.domain.categories_count - 1; ++profile_index) {
      shuffle<unsigned>(random, ref(criterion_indexes));
      improve_model_profile(random, models, model_index, profile_index, criterion_indexes);
    }
  }

 public:
  void improve_profiles(std::shared_ptr<Models> models) override {
    #pragma omp parallel for
    for (unsigned model_index = 0; model_index != models->models_count; ++model_index) {
      improve_model_profiles(_random, *models, model_index);
    }
  }

 private:
  const Random& _random;
};

class OptimizeWeightsUsingGlop : public WeightsOptimizationStrategy {
  struct LinearProgram {
    std::shared_ptr<glp::LinearProgram> program;
    std::vector<glp::ColIndex> weight_variables;
    std::vector<glp::ColIndex> x_variables;
    std::vector<glp::ColIndex> xp_variables;
    std::vector<glp::ColIndex> y_variables;
    std::vector<glp::ColIndex> yp_variables;
  };

  std::shared_ptr<LinearProgram> make_internal_linear_program(
    const float epsilon,
    const Models& models,
    unsigned model_index
  ) {
    auto lp = std::make_shared<LinearProgram>();

    lp->program = std::make_shared<glp::LinearProgram>();
    lp->weight_variables.reserve(models.domain.criteria_count);
    for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      lp->weight_variables.push_back(lp->program->CreateNewVariable());
    }

    lp->x_variables.reserve(models.domain.learning_alternatives_count);
    lp->xp_variables.reserve(models.domain.learning_alternatives_count);
    lp->y_variables.reserve(models.domain.learning_alternatives_count);
    lp->yp_variables.reserve(models.domain.learning_alternatives_count);
    for (unsigned alt_index = 0; alt_index != models.domain.learning_alternatives_count; ++alt_index) {
      lp->x_variables.push_back(lp->program->CreateNewVariable());
      lp->xp_variables.push_back(lp->program->CreateNewVariable());
      lp->y_variables.push_back(lp->program->CreateNewVariable());
      lp->yp_variables.push_back(lp->program->CreateNewVariable());

      lp->program->SetObjectiveCoefficient(lp->xp_variables.back(), 1);
      lp->program->SetObjectiveCoefficient(lp->yp_variables.back(), 1);

      const unsigned category_index = models.domain.learning_assignments[alt_index];

      if (category_index != 0) {
        glp::RowIndex c = lp->program->CreateNewConstraint();
        lp->program->SetConstraintBounds(c, 1, 1);
        lp->program->SetCoefficient(c, lp->x_variables.back(), -1);
        lp->program->SetCoefficient(c, lp->xp_variables.back(), 1);
        for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
          const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
          const float profile_value = models.profiles[crit_index][category_index - 1][model_index];
          if (alternative_value >= profile_value) {
            lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
          }
        }
      }

      if (category_index != models.domain.categories_count - 1) {
        glp::RowIndex c = lp->program->CreateNewConstraint();
        lp->program->SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
        lp->program->SetCoefficient(c, lp->y_variables.back(), 1);
        lp->program->SetCoefficient(c, lp->yp_variables.back(), -1);
        for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
          const float alternative_value = models.domain.learning_alternatives[crit_index][alt_index];
          const float profile_value = models.profiles[crit_index][category_index][model_index];
          if (alternative_value >= profile_value) {
            lp->program->SetCoefficient(c, lp->weight_variables[crit_index], 1);
          }
        }
      }
    }

    return lp;
  }

  auto solve_linear_program(std::shared_ptr<LinearProgram> lp) {
    operations_research::glop::LPSolver solver;
    operations_research::glop::GlopParameters parameters;
    parameters.set_provide_strong_optimal_guarantee(true);
    solver.SetParameters(parameters);

    auto status = solver.Solve(*lp->program);
    assert(status == operations_research::glop::ProblemStatus::OPTIMAL);
    auto values = solver.variable_values();

    return values;
  }

  void optimize_weights(const Models& models, unsigned model_index) {
    auto lp = make_internal_linear_program(1e-6, models, model_index);
    auto values = solve_linear_program(lp);

    for (unsigned crit_index = 0; crit_index != models.domain.criteria_count; ++crit_index) {
      models.weights[crit_index][model_index] = values[lp->weight_variables[crit_index]];
    }
  }

  void optimize_weights(const Models& models) {
    #pragma omp parallel for
    for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
      optimize_weights(models, model_index);
    }
  }

 public:
  void optimize_weights(std::shared_ptr<Models> models) override {
    optimize_weights(*models);
  };
};

class TerminateAtAccuracy : public TerminationStrategy {
 public:
  explicit TerminateAtAccuracy(unsigned target_accuracy) :
    _target_accuracy(target_accuracy) {}

  bool terminate(unsigned /*iteration_index*/, unsigned best_accuracy) override {
    return best_accuracy >= _target_accuracy;
  }

 private:
  unsigned _target_accuracy;
};

std::vector<unsigned> partition_models_by_accuracy(const unsigned models_count, const Models& models) {
  std::vector<unsigned> accuracies(models_count, 0);
  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    accuracies[model_index] = get_accuracy(models, model_index);
  }

  std::vector<unsigned> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [&accuracies](unsigned left_model_index, unsigned right_model_index) {
      return accuracies[left_model_index] < accuracies[right_model_index];
    });

  return model_indexes;
}

io::Model perform_learning(
  std::shared_ptr<Models> models,
  std::shared_ptr<ProfilesInitializationStrategy> profiles_initialization_strategy,
  std::shared_ptr<WeightsOptimizationStrategy> weights_optimization_strategy,
  std::shared_ptr<ProfilesImprovementStrategy> profiles_improvement_strategy,
  std::shared_ptr<TerminationStrategy> termination_strategy
) {
  const unsigned models_count = models->models_count;

  std::vector<unsigned> model_indexes(models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  profiles_initialization_strategy->initialize_profiles(
    models,
    0,
    model_indexes.begin(), model_indexes.end());

  unsigned best_accuracy = 0;

  for (int iteration_index = 0; !termination_strategy->terminate(iteration_index, best_accuracy); ++iteration_index) {
    if (iteration_index != 0) {
      profiles_initialization_strategy->initialize_profiles(
        models,
        iteration_index,
        model_indexes.begin(), model_indexes.begin() + models_count / 2);
    }

    weights_optimization_strategy->optimize_weights(models);
    profiles_improvement_strategy->improve_profiles(models);

    model_indexes = partition_models_by_accuracy(models_count, *models);
    best_accuracy = get_accuracy(*models, model_indexes.back());
  }

  return models->unmake_one(model_indexes.back());
}

struct MrSortLearning_ {
  const Domain& domain;
  const Alternatives& learning_set;

  Model perform();
};

Model MrSortLearning_::perform() {
  std::map<std::string, unsigned> category_indexes;
  for (const auto& category: domain.categories) {
    category_indexes[category.name] = category_indexes.size();
  }

  std::vector<io::ClassifiedAlternative> ppl_alternatives;
  for (const auto& alt : learning_set.alternatives) {
    ppl_alternatives.emplace_back(
      alt.profile,
      category_indexes[*alt.category]
    );
  }
  ppl_alternatives.reserve(learning_set.alternatives.size());
  io::LearningSet ppl_learning_set(
    domain.criteria.size(),
    domain.categories.size(),
    ppl_alternatives.size(),
    ppl_alternatives
  );

  Random random(44);
  auto ppl_host_domain = Domain_::make(ppl_learning_set);
  auto ppl_host_models = Models::make(*ppl_host_domain, default_models_count);
  io::Model ppl_model = perform_learning(
    ppl_host_models,
    std::make_shared<InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion>(random, *ppl_host_models),
    std::make_shared<OptimizeWeightsUsingGlop>(),
    std::make_shared<ImproveProfilesWithAccuracyHeuristic>(random),
    std::make_shared<TerminateAtAccuracy>(learning_set.alternatives.size()));

  Model::SufficientCoalitions coalitions{
    Model::SufficientCoalitions::Kind::weights,
    ppl_model.weights,
  };
  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(domain.categories.size() - 1);
  assert(ppl_model.profiles.size() == domain.categories.size() - 1);
  for (const auto& profile: ppl_model.profiles) {
    boundaries.emplace_back(profile, coalitions);
  }
  return Model{domain, boundaries};
}

Model MrSortLearning::perform() {
  return MrSortLearning_{
    domain,
    learning_set,
  }.perform();
}

TEST_CASE("Basic MR-Sort learning") {
  Domain domain = Domain::generate(3, 2, 41);
  Model model = Model::generate_mrsort(domain, 42);
  Alternatives learning_set = Alternatives::generate(domain, model, 100, 43);

  Model learned_model = MrSortLearning{domain, learning_set}.perform();

  {
    ClassificationResult result = classify_alternatives(domain, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 100);
  }

  {
    Alternatives testing_set = Alternatives::generate(domain, model, 1000, 43);
    ClassificationResult result = classify_alternatives(domain, learned_model, &testing_set);
    CHECK(result.changed == 3);
    CHECK(result.unchanged == 997);
  }
}

}  // namespace lincs
