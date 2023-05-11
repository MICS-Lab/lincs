#include "lincs.hpp"

#include <magic_enum.hpp>
#include <omp.h>
#include <ortools/glop/lp_solver.h>

#include "median-and-max.hpp"

#include <doctest.h>  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace glp = operations_research::glop;

namespace lincs {

WeightsProfilesBreedMrSortLearning::Models WeightsProfilesBreedMrSortLearning::Models::make(const Domain& domain, const Alternatives& learning_set, const unsigned models_count, const unsigned random_seed) {
  std::map<std::string, unsigned> category_indexes;
  for (const auto& category: domain.categories) {
    category_indexes[category.name] = category_indexes.size();
  }

  const unsigned criteria_count = domain.criteria.size();
  const unsigned categories_count = domain.categories.size();
  const unsigned alternatives_count = learning_set.alternatives.size();

  Array2D<Host, float> alternatives(criteria_count, alternatives_count, uninitialized);
  Array1D<Host, unsigned> assignments(alternatives_count, uninitialized);

  for (unsigned alternative_index = 0; alternative_index != alternatives_count; ++alternative_index) {
    const Alternative& alt = learning_set.alternatives[alternative_index];

    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      alternatives[criterion_index][alternative_index] = alt.profile[criterion_index];
    }

    assignments[alternative_index] = category_indexes[*alt.category];
  }

  Array2D<Host, float> weights(criteria_count, models_count, uninitialized);
  Array3D<Host, float> profiles(criteria_count, (categories_count - 1), models_count, uninitialized);

  std::vector<std::mt19937> urbgs(models_count);
  for (unsigned model_index = 0; model_index != models_count; ++model_index) {
    urbgs[model_index].seed(random_seed * (model_index + 1));
  }

  return {
    domain,
    categories_count,
    criteria_count,
    alternatives_count,
    std::move(alternatives),
    std::move(assignments),
    models_count,
    std::move(weights),
    std::move(profiles),
    std::move(urbgs),
  };
}

Model WeightsProfilesBreedMrSortLearning::Models::get_model(const unsigned model_index) const {
  assert(model_index < models_count);

  std::vector<float> model_weights;
  model_weights.reserve(criteria_count);
  for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
    model_weights.push_back(weights[criterion_index][model_index]);
  }
  Model::SufficientCoalitions coalitions{Model::SufficientCoalitions::Kind::weights, model_weights};

  std::vector<Model::Boundary> boundaries;
  boundaries.reserve(categories_count - 1);
  for (unsigned cat_index = 0; cat_index != categories_count - 1; ++cat_index) {
    std::vector<float> boundary_profile;
    boundary_profile.reserve(criteria_count);
    for (unsigned criterion_index = 0; criterion_index != criteria_count; ++criterion_index) {
      boundary_profile.push_back(profiles[criterion_index][cat_index][model_index]);
    }
    boundaries.emplace_back(boundary_profile, coalitions);
  }

  return Model{domain, boundaries};
}

Model WeightsProfilesBreedMrSortLearning::perform() {
  std::vector<unsigned> model_indexes(models.models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  profiles_initialization_strategy.initialize_profiles(model_indexes.begin(), model_indexes.end());

  unsigned best_accuracy = 0;

  for (int iteration_index = 0; !termination_strategy.terminate(iteration_index, best_accuracy); ++iteration_index) {
    if (iteration_index != 0) {
      profiles_initialization_strategy.initialize_profiles(model_indexes.begin(), model_indexes.begin() + models.models_count / 2);
    }

    weights_optimization_strategy.optimize_weights();
    profiles_improvement_strategy.improve_profiles();

    auto p = partition_models_by_accuracy();
    model_indexes = std::move(p.first);
    best_accuracy = p.second;
  }

  return models.get_model(model_indexes.back());
}

std::pair<std::vector<unsigned>, unsigned> WeightsProfilesBreedMrSortLearning::partition_models_by_accuracy() {
  std::vector<unsigned> accuracies(models.models_count, 0);
  for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
    accuracies[model_index] = get_accuracy(model_index);
  }

  std::vector<unsigned> model_indexes(models.models_count, 0);
  std::iota(model_indexes.begin(), model_indexes.end(), 0);
  ensure_median_and_max(
    model_indexes.begin(), model_indexes.end(),
    [&accuracies](unsigned left_model_index, unsigned right_model_index) {
      return accuracies[left_model_index] < accuracies[right_model_index];
    });

  return std::make_pair(model_indexes, accuracies[model_indexes.back()]);
}

unsigned WeightsProfilesBreedMrSortLearning::get_accuracy(const unsigned model_index) {
  unsigned accuracy = 0;

  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    if (is_correctly_assigned(model_index, alternative_index)) {
      ++accuracy;
    }
  }

  return accuracy;
}

bool WeightsProfilesBreedMrSortLearning::is_correctly_assigned(
    const unsigned model_index,
    const unsigned alternative_index) {
  const unsigned expected_assignment = models.learning_assignments[alternative_index];
  const unsigned actual_assignment = get_assignment(models, model_index, alternative_index);

  return actual_assignment == expected_assignment;
}

unsigned WeightsProfilesBreedMrSortLearning::get_assignment(const Models& models, const unsigned model_index, const unsigned alternative_index) {
  // @todo Evaluate if it's worth storing and updating the models' assignments
  // (instead of recomputing them here)
  assert(model_index < models.models_count);
  assert(alternative_index < models.learning_alternatives_count);

  // Not parallelizable in this form because the loop gets interrupted by a return. But we could rewrite it
  // to always perform all its iterations, and then it would be yet another map-reduce, with the reduce
  // phase keeping the maximum 'category_index' that passes the weight threshold.
  for (unsigned category_index = models.categories_count - 1; category_index != 0; --category_index) {
    const unsigned profile_index = category_index - 1;
    float weight_at_or_above_profile = 0;
    for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
      const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
      const float profile_value = models.profiles[criterion_index][profile_index][model_index];
      if (alternative_value >= profile_value) {
        weight_at_or_above_profile += models.weights[criterion_index][model_index];
      }
    }
    if (weight_at_or_above_profile >= 1) {
      return category_index;
    }
  }
  return 0;
}


InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion(Models& models_) : models(models_) {
  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    generators.emplace_back();
    for (unsigned profile_index = 0; profile_index != models.categories_count - 1; ++profile_index) {
      generators.back().emplace_back(get_candidate_probabilities(criterion_index, profile_index));
    }
  }
}

std::map<float, double> InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::get_candidate_probabilities(
  unsigned criterion_index,
  unsigned profile_index
) {
  std::vector<float> values_below;
  // The size used for 'reserve' is a few times larger than the actual final size,
  // so we're allocating too much memory. As it's temporary, I don't think it's too bad.
  // If 'initialize' ever becomes the centre of focus for our optimization effort, we should measure.
  values_below.reserve(models.learning_alternatives_count);
  std::vector<float> values_above;
  values_above.reserve(models.learning_alternatives_count);
  // This loop could/should be done once outside this function
  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    const float value = models.learning_alternatives[criterion_index][alternative_index];
    const unsigned assignment = models.learning_assignments[alternative_index];
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

void InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion::initialize_profiles(
  std::vector<unsigned>::const_iterator model_indexes_begin,
  const std::vector<unsigned>::const_iterator model_indexes_end
) {
  // Embarrassingly parallel
  for (; model_indexes_begin != model_indexes_end; ++model_indexes_begin) {
    const unsigned model_index = *model_indexes_begin;

    // Embarrassingly parallel
    for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
      // Not parallel because of the profiles ordering constraint
      for (unsigned category_index = models.categories_count - 1; category_index != 0; --category_index) {
        const unsigned profile_index = category_index - 1;
        float value = generators[criterion_index][profile_index](models.urbgs[model_index]);

        if (profile_index != models.categories_count - 2) {
          value = std::min(value, models.profiles[criterion_index][profile_index + 1][model_index]);
        }
        // @todo Add a unit test that triggers the following assertion
        // (This will require removing the code to enforce the order of profiles above)
        // Then restore the code to enforce the order of profiles
        // Note, this assertion does not protect us from initializing a model with two identical profiles.
        // Is it really that bad?
        assert(
          profile_index == models.categories_count - 2
          || models.profiles[criterion_index][profile_index + 1][model_index] >= value);

        models.profiles[criterion_index][profile_index][model_index] = value;
      }
    }
  }
}


void OptimizeWeightsUsingGlop::optimize_weights() {
  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
    optimize_model_weights(model_index);
  }
};

struct OptimizeWeightsUsingGlop::LinearProgram {
  std::shared_ptr<glp::LinearProgram> program;
  std::vector<glp::ColIndex> weight_variables;
  std::vector<glp::ColIndex> x_variables;
  std::vector<glp::ColIndex> xp_variables;
  std::vector<glp::ColIndex> y_variables;
  std::vector<glp::ColIndex> yp_variables;
};

auto OptimizeWeightsUsingGlop::solve_linear_program(std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> lp) {
  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  auto status = solver.Solve(*lp->program);
  assert(status == operations_research::glop::ProblemStatus::OPTIMAL);
  auto values = solver.variable_values();

  return values;
}

void OptimizeWeightsUsingGlop::optimize_model_weights(unsigned model_index) {
  auto lp = make_internal_linear_program(1e-6, model_index);
  auto values = solve_linear_program(lp);

  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    models.weights[criterion_index][model_index] = values[lp->weight_variables[criterion_index]];
  }
}

std::shared_ptr<OptimizeWeightsUsingGlop::LinearProgram> OptimizeWeightsUsingGlop::make_internal_linear_program(
  const float epsilon,
  unsigned model_index
) {
  auto lp = std::make_shared<LinearProgram>();

  lp->program = std::make_shared<glp::LinearProgram>();
  lp->weight_variables.reserve(models.criteria_count);
  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    lp->weight_variables.push_back(lp->program->CreateNewVariable());
  }

  lp->x_variables.reserve(models.learning_alternatives_count);
  lp->xp_variables.reserve(models.learning_alternatives_count);
  lp->y_variables.reserve(models.learning_alternatives_count);
  lp->yp_variables.reserve(models.learning_alternatives_count);
  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    lp->x_variables.push_back(lp->program->CreateNewVariable());
    lp->xp_variables.push_back(lp->program->CreateNewVariable());
    lp->y_variables.push_back(lp->program->CreateNewVariable());
    lp->yp_variables.push_back(lp->program->CreateNewVariable());

    lp->program->SetObjectiveCoefficient(lp->xp_variables.back(), 1);
    lp->program->SetObjectiveCoefficient(lp->yp_variables.back(), 1);

    const unsigned category_index = models.learning_assignments[alternative_index];

    if (category_index != 0) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1, 1);
      lp->program->SetCoefficient(c, lp->x_variables.back(), -1);
      lp->program->SetCoefficient(c, lp->xp_variables.back(), 1);
      for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
        const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = models.profiles[criterion_index][category_index - 1][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[criterion_index], 1);
        }
      }
    }

    if (category_index != models.categories_count - 1) {
      glp::RowIndex c = lp->program->CreateNewConstraint();
      lp->program->SetConstraintBounds(c, 1 - epsilon, 1 - epsilon);
      lp->program->SetCoefficient(c, lp->y_variables.back(), 1);
      lp->program->SetCoefficient(c, lp->yp_variables.back(), -1);
      for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
        const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
        const float profile_value = models.profiles[criterion_index][category_index][model_index];
        if (alternative_value >= profile_value) {
          lp->program->SetCoefficient(c, lp->weight_variables[criterion_index], 1);
        }
      }
    }
  }

  return lp;
}


void ImproveProfilesWithAccuracyHeuristic::improve_profiles() {
  #pragma omp parallel for
  for (unsigned model_index = 0; model_index != models.models_count; ++model_index) {
    improve_model_profiles(model_index);
  }
}

void ImproveProfilesWithAccuracyHeuristic::improve_model_profiles(const unsigned model_index) {
  Array1D<Host, unsigned> criterion_indexes(models.criteria_count, uninitialized);
  // Not worth parallelizing because models.criteria_count is typically small
  for (unsigned crit_idx_idx = 0; crit_idx_idx != models.criteria_count; ++crit_idx_idx) {
    criterion_indexes[crit_idx_idx] = crit_idx_idx;
  }

  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned profile_index = 0; profile_index != models.categories_count - 1; ++profile_index) {
    shuffle<unsigned>(model_index, ref(criterion_indexes));
    improve_model_profile(model_index, profile_index, criterion_indexes);
  }
}

void ImproveProfilesWithAccuracyHeuristic::improve_model_profile(
  const unsigned model_index,
  const unsigned profile_index,
  ArrayView1D<Host, const unsigned> criterion_indexes
) {
  // Not parallel because iteration N+1 relies on side effect in iteration N
  // (We could challenge this aspect of the algorithm described by Sobrie)
  for (unsigned crit_idx_idx = 0; crit_idx_idx != models.criteria_count; ++crit_idx_idx) {
    improve_model_profile(model_index, profile_index, criterion_indexes[crit_idx_idx]);
  }
}

void ImproveProfilesWithAccuracyHeuristic::improve_model_profile(
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
    profile_index == models.categories_count - 2 ? 1. :
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
    float destination = highest_destination;
    // By specification, std::uniform_real_distribution should never return its highest value,
    // but "most existing implementations have a bug where they may occasionally" return it,
    // so we work around that bug by calling it again until it doesn't.
    // Ref: https://en.cppreference.com/w/cpp/numeric/random/uniform_real_distribution
    while (destination == highest_destination) {
      destination = std::uniform_real_distribution<float>(lowest_destination, highest_destination)(models.urbgs[model_index]);
    }
    const float desirability = compute_move_desirability(
      models, model_index, profile_index, criterion_index, destination).value();
    // Single-key reduce (divide and conquer?) (atomic compare-and-swap?)
    if (desirability > best_desirability) {
      best_desirability = desirability;
      best_destination = destination;
    }
  }

  // @todo Desirability can be as high as 2. The [0, 1] interval is a weird choice.
  if (std::uniform_real_distribution<float>(0, 1)(models.urbgs[model_index]) <= best_desirability) {
    models.profiles[criterion_index][profile_index][model_index] = best_destination;
  }
}

ImproveProfilesWithAccuracyHeuristic::Desirability ImproveProfilesWithAccuracyHeuristic::compute_move_desirability(
  const Models& models,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const float destination
) {
  Desirability d;

  for (unsigned alternative_index = 0; alternative_index != models.learning_alternatives_count; ++alternative_index) {
    update_move_desirability(
      models, model_index, profile_index, criterion_index, destination, alternative_index, &d);
  }

  return d;
}

void ImproveProfilesWithAccuracyHeuristic::update_move_desirability(
  const Models& models,
  const unsigned model_index,
  const unsigned profile_index,
  const unsigned criterion_index,
  const float destination,
  const unsigned alternative_index,
  Desirability* desirability
) {
  const float current_position = models.profiles[criterion_index][profile_index][model_index];
  const float weight = models.weights[criterion_index][model_index];

  const float value = models.learning_alternatives[criterion_index][alternative_index];
  const unsigned learning_assignment = models.learning_assignments[alternative_index];
  const unsigned model_assignment = WeightsProfilesBreedMrSortLearning::get_assignment(models, model_index, alternative_index);

  // @todo Factorize with get_assignment
  float weight_at_or_above_profile = 0;
  for (unsigned criterion_index = 0; criterion_index != models.criteria_count; ++criterion_index) {
    const float alternative_value = models.learning_alternatives[criterion_index][alternative_index];
    const float profile_value = models.profiles[criterion_index][profile_index][model_index];
    if (alternative_value >= profile_value) {
      weight_at_or_above_profile += models.weights[criterion_index][model_index];
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

float ImproveProfilesWithAccuracyHeuristic::Desirability::value() const {
  if (v + w + t + q + r == 0) {
    return zero_value;
  } else {
    return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
  }
}


bool TerminateAtAccuracy::terminate(unsigned /*iteration_index*/, unsigned best_accuracy) {
  return best_accuracy >= _target_accuracy;
}

TEST_CASE("Basic MR-Sort learning") {
  Domain domain = Domain::generate(3, 2, 41);
  Model model = Model::generate_mrsort(domain, 42);
  Alternatives learning_set = Alternatives::generate(domain, model, 100, 43);

  const unsigned random_seed = 44;
  auto models = WeightsProfilesBreedMrSortLearning::Models::make(
    domain, learning_set, WeightsProfilesBreedMrSortLearning::default_models_count, random_seed);

  InitializeProfilesForProbabilisticMaximalDiscriminationPowerPerCriterion profiles_initialization_strategy(models);
  OptimizeWeightsUsingGlop weights_optimization_strategy(models);
  ImproveProfilesWithAccuracyHeuristic profiles_improvement_strategy(models);
  TerminateAtAccuracy termination_strategy(learning_set.alternatives.size());

  Model learned_model = WeightsProfilesBreedMrSortLearning(
    models,
    profiles_initialization_strategy,
    weights_optimization_strategy,
    profiles_improvement_strategy,
    termination_strategy
  ).perform();

  {
    ClassificationResult result = classify_alternatives(domain, learned_model, &learning_set);
    CHECK(result.changed == 0);
    CHECK(result.unchanged == 100);
  }

  {
    Alternatives testing_set = Alternatives::generate(domain, model, 1000, 43);
    ClassificationResult result = classify_alternatives(domain, learned_model, &testing_set);
    CHECK(result.changed == 6);
    CHECK(result.unchanged == 994);
  }
}

}  // namespace lincs
