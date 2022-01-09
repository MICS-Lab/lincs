// Copyright 2021-2022 Vincent Jacques

#include <chrones.hpp>

#include "../assign.hpp"
#include "../generate.hpp"
#include "../test-utils.hpp"
#include "desirability.hpp"
#include "heuristic-for-accuracy-random-candidates.hpp"


CHRONABLE("heuristic-for-accuracy-random-candidates-tests")

namespace ppl {

TEST(ImproveProfiles, First) {
  auto host_domain = make_domain(2, {{{0.5}, 0}});

  auto make_host_models = [&host_domain]() { return make_models(host_domain, {{{{0.1}}, {1}}}); };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(*host_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*host_models, 0), 1);
  }

  {
    auto host_models = make_host_models();
    auto device_models = host_models->clone_to<Device>(host_domain->clone_to<Device>());

    EXPECT_EQ(get_accuracy(*device_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, device_models).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 1);
  }
}

TEST(ImproveProfiles, SingleCriterion) {
  std::mt19937 gen(42);
  auto model = generate::model(&gen, 1, 2);
  model.weights.front() = 1;

  auto learning_set = generate::learning_set(&gen, model, 25);
  auto host_domain = Domain<Host>::make(learning_set);

  auto make_host_models = [&host_domain]() { return make_models(host_domain, {{{{0}}, {1}}}); };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(*host_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*host_models, 0), 23);
  }

  {
    auto host_models = make_host_models();
    auto device_models = host_models->clone_to<Device>(host_domain->clone_to<Device>());

    EXPECT_EQ(get_accuracy(*device_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, device_models).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 23);
  }
}

TEST(ImproveProfiles, Larger) {
  std::mt19937 gen(42);
  auto model = generate::model(&gen, 4, 5);

  std::fill(model.weights.begin(), model.weights.end(), 0.5);

  auto learning_set = generate::learning_set(&gen, model, 250);
  auto host_domain = Domain<Host>::make(learning_set);

  auto make_host_models = [&host_domain]() {
    return make_models(
      host_domain,
      {{
        {
          {0.2, 0.2, 0.2, 0.2},
          {0.4, 0.4, 0.4, 0.4},
          {0.6, 0.6, 0.6, 0.6},
          {0.8, 0.8, 0.8, 0.8}},
        {0.5, 0.5, 0.5, 0.5}}});
  };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(*host_models, 0), 132);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*host_models, 0), 164);
  }

  {
    auto host_models = make_host_models();
    auto device_models = host_models->clone_to<Device>(host_domain->clone_to<Device>());

    EXPECT_EQ(get_accuracy(*device_models, 0), 132);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, device_models).improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 163);
  }
}

}  // namespace ppl
