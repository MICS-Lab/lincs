#include "lincs.hpp"

#include <fstream>
#include <chrono>

#include <doctest.h>

#include "ppl.hpp"

namespace ppl {

void ImproveProfilesWithAccuracyHeuristicOnGpu::improve_profiles(std::shared_ptr<Models<Host>> host_models) {}

}  // namespace ppl

std::vector<std::shared_ptr<ppl::LearningObserver>> make_observers(
  const bool quiet,
  std::optional<std::ofstream>& intermediate_models_file
);

std::shared_ptr<ppl::ProfilesInitializationStrategy> make_profiles_initialization_strategy(
  const Random& random,
  const ppl::Models<Host>& models
);

enum class WeightsOptimizationStrategy {
  glop,
  glop_reuse,
};

std::shared_ptr<ppl::WeightsOptimizationStrategy> make_weights_optimization_strategy(
  WeightsOptimizationStrategy strategy,
  const ppl::Models<Host>& host_models
);

enum class ProfilesImprovementStrategy {
  heuristic,
};

std::shared_ptr<ppl::ProfilesImprovementStrategy> make_profiles_improvement_strategy(
  ProfilesImprovementStrategy strategy,
  const bool use_gpu,
  const Random& random,
  std::shared_ptr<ppl::Domain<Host>> domain,
  std::shared_ptr<ppl::Models<Host>> models
);

std::shared_ptr<ppl::TerminationStrategy> make_termination_strategy(
  uint target_accuracy,
  std::optional<uint> max_iterations,
  std::optional<std::chrono::seconds> max_duration
);

namespace lincs {

Model MrSortLearning::perform() {
  std::map<std::string, unsigned> category_indexes;
  for (const auto& category: domain.categories) {
    category_indexes[category.name] = category_indexes.size();
  }

  std::vector<ppl::io::ClassifiedAlternative> ppl_alternatives;
  for (const auto& alt : learning_set.alternatives) {
    ppl_alternatives.emplace_back(
      alt.profile,
      category_indexes[*alt.category]
    );
  }
  ppl_alternatives.reserve(learning_set.alternatives.size());
  ppl::io::LearningSet ppl_learning_set(
    domain.criteria.size(),
    domain.categories.size(),
    ppl_alternatives.size(),
    ppl_alternatives
  );

  const unsigned random_seed = 44;
  const unsigned models_count = ppl::default_models_count;
  const bool quiet = true;
  /* const */ std::optional<std::ofstream> intermediate_models_file;
  const WeightsOptimizationStrategy weights_optimization_strategy = WeightsOptimizationStrategy::glop;
  const ProfilesImprovementStrategy profiles_improvement_strategy = ProfilesImprovementStrategy::heuristic;
  const bool use_gpu = false;
  const uint target_accuracy = learning_set.alternatives.size();
  const std::optional<uint> max_iterations;
  const std::optional<std::chrono::seconds> max_duration;

  Random random(random_seed);
  auto ppl_host_domain = ppl::Domain<Host>::make(ppl_learning_set);
  auto ppl_host_models = ppl::Models<Host>::make(ppl_host_domain, models_count);
  auto ppl_result = ppl::perform_learning(
    ppl_host_models,
    make_observers(quiet, intermediate_models_file),
    make_profiles_initialization_strategy(random, *ppl_host_models),
    make_weights_optimization_strategy(weights_optimization_strategy, *ppl_host_models),
    make_profiles_improvement_strategy(profiles_improvement_strategy, use_gpu, random, ppl_host_domain, ppl_host_models),
    make_termination_strategy(target_accuracy, max_iterations, max_duration));
  ppl::io::Model ppl_model = ppl_result.best_model;

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
