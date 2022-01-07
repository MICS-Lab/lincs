// Copyright 2021-2022 Vincent Jacques

#include <chrones.hpp>

#include "../assign.hpp"
#include "../generate.hpp"
#include "../test-utils.hpp"
#include "heuristic-for-accuracy-random-candidates.hpp"


CHRONABLE("heuristic-for-accuracy-random-candidates-tests")

namespace ppl {

// Internal function (not declared in the header) that we still want to unit-test
__host__ __device__ Desirability compute_move_desirability(
  const ModelsView&,
  uint model_index,
  uint profile_index,
  uint criterion_index,
  float destination);

Desirability compute_move_desirability(
    const Models<Host>& models,
    uint model_index,
    uint profile_index,
    uint criterion_index,
    float destination) {
  return compute_move_desirability(
    models.get_view(), model_index, profile_index, criterion_index, destination);
}


TEST(ComputeMoveDesirability, NoImpact) {
  auto domain = make_domain(2, {{{0.5}, 0}});
  auto models = make_models(domain, {{{{0.1}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.2);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveUpForOneMoreCorrectAssignment) {
  auto domain = make_domain(2, {{{0.5}, 0}});
  auto models = make_models(domain, {{{{0.4}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.6);
  EXPECT_EQ(d.v, 1);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveDownForOneMoreCorrectAssignment) {
  auto domain = make_domain(2, {{{0.5}, 1}});
  auto models = make_models(domain, {{{{0.6}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.4);
  EXPECT_EQ(d.v, 1);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveUpForIncreasedCorrectCoalition) {
  auto domain = make_domain(2, {{{0.5, 0.5}, 0}});
  auto models = make_models(domain, {{{{0.4, 0.4}}, {1, 1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.6);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 1);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveDownForIncreasedCorrectCoalition) {
  auto domain = make_domain(2, {{{0.5, 0.5}, 1}});
  auto models = make_models(domain, {{{{0.6, 0.6}}, {0.5, 0.5}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.4);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 1);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveUpForOneFewerCorrectAssignment) {
  auto domain = make_domain(2, {{{0.5}, 1}});
  auto models = make_models(domain, {{{{0.4}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.6);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 1);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveDownForOneFewerCorrectAssignment) {
  auto domain = make_domain(2, {{{0.5}, 0}});
  auto models = make_models(domain, {{{{0.6}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.4);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 1);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveUpForDecreasedCorrectCoalition) {
  auto domain = make_domain(2, {{{0.5, 0.5}, 1}});
  auto models = make_models(domain, {{{{0.4, 0.6}}, {0.5, 0.5}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.6);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 1);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveDownForDecreasedCorrectCoalition) {
  auto domain = make_domain(2, {{{0.5, 0.5}, 0}});
  auto models = make_models(domain, {{{{0.6, 0.4}}, {1, 1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.4);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 1);
  EXPECT_EQ(d.t, 0);
}

TEST(ComputeMoveDesirability, MoveUpForIncreasedBetterCoalition) {
  auto domain = make_domain(3, {{{0.5}, 0}});
  auto models = make_models(domain, {{{{0.3}, {0.4}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 1, 0, 0.6);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 1);
}

TEST(ComputeMoveDesirability, MoveDownForIncreasedBetterCoalition) {
  auto domain = make_domain(3, {{{0.5}, 2}});
  auto models = make_models(domain, {{{{0.6}, {0.7}}, {1}}});

  Desirability d = compute_move_desirability(models, 0, 0, 0, 0.4);
  EXPECT_EQ(d.v, 0);
  EXPECT_EQ(d.w, 0);
  EXPECT_EQ(d.q, 0);
  EXPECT_EQ(d.r, 0);
  EXPECT_EQ(d.t, 1);
}

TEST(ImproveProfiles, First) {
  auto host_domain = make_domain(2, {{{0.5}, 0}});
  auto make_host_models = [&host_domain]() { return make_models(host_domain, {{{{0.1}}, {1}}}); };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(host_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(host_models, 0), 1);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain.clone_to<Device>();
    auto device_models = host_models.clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(device_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, &device_models).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(device_models, 0), 1);
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

    EXPECT_EQ(get_accuracy(host_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(host_models, 0), 23);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain.clone_to<Device>();
    auto device_models = host_models.clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(device_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, &device_models).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(device_models, 0), 23);
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

    EXPECT_EQ(get_accuracy(host_models, 0), 132);
    ImproveProfilesWithAccuracyHeuristicOnCpu(random).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(host_models, 0), 164);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain.clone_to<Device>();
    auto device_models = host_models.clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(device_models, 0), 132);
    ImproveProfilesWithAccuracyHeuristicOnGpu(random, &device_models).improve_profiles(&host_models);
    EXPECT_EQ(get_accuracy(device_models, 0), 163);
  }
}

}  // namespace ppl
