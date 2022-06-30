// Copyright 2021-2022 Vincent Jacques

#include <chrones.hpp>

#include "../assign.hpp"
#include "../generate.hpp"
#include "../test-utils.hpp"
#include "desirability.hpp"
#include "heuristic-for-accuracy-midpoint-candidates.hpp"


CHRONABLE("heuristic-for-accuracy-midpoint-candidates-tests")

namespace ppl {

// Internal functions (not declared in the header) that we still want to unit-test
__host__ __device__
uint find_smallest_index_above(const ArrayView1D<Anywhere, const float>&, const uint, const float);

__host__ __device__
uint find_greatest_index_below(const ArrayView1D<Anywhere, const float>&, const uint, const float);

TEST(FindIndex, One) {
  float data[] {1, 42};
  ArrayView1D<Host, float> m(2, data);

  EXPECT_EQ(find_greatest_index_below(m, 1, 1), 0);
  EXPECT_EQ(find_smallest_index_above(m, 1, 1), 0);
}

TEST(FindIndex, Several) {
  float data[] {1, 2, 3, 4, 5, 42};
  ArrayView1D<Host, float> m(6, data);

  EXPECT_EQ(find_greatest_index_below(m, 5, 3.2), 2);
  EXPECT_EQ(find_smallest_index_above(m, 5, 3.2), 3);

  EXPECT_EQ(find_greatest_index_below(m, 5, 1), 0);
  EXPECT_EQ(find_smallest_index_above(m, 5, 1), 0);

  EXPECT_EQ(find_greatest_index_below(m, 5, 5), 4);
  EXPECT_EQ(find_smallest_index_above(m, 5, 5), 4);
}

TEST(ImproveProfiles, First) {
  auto host_domain = make_domain(2, {{{0.5}, 0}});
  auto host_candidates = Candidates<Host>::make(host_domain);

  auto make_host_models = [&host_domain]() { return make_models(host_domain, {{{{0.1}}, {1}}}); };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(*host_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu(random, host_candidates)
      .improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*host_models, 0), 1);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain->clone_to<Device>();
    auto device_models = host_models->clone_to<Device>(device_domain);
    auto device_candidates = host_candidates->clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(*device_models, 0), 0);
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu(random, device_models, device_candidates)
      .improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 1);
  }
}

TEST(ImproveProfiles, SingleCriterion) {
  std::mt19937 gen(42);
  auto model = generate::model(&gen, 1, 2);
  model.weights.front() = 1;

  auto learning_set = generate::learning_set(&gen, model, 25);
  auto host_domain = Domain<Host>::make(learning_set);
  auto host_candidates = Candidates<Host>::make(host_domain);

  auto make_host_models = [&host_domain]() { return make_models(host_domain, {{{{0}}, {1}}}); };

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  {
    auto host_models = make_host_models();

    EXPECT_EQ(get_accuracy(*host_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu(random, host_candidates)
      .improve_profiles(host_models);
    // @todo Why are all accuracies lower than for the "random candidates" version?
    EXPECT_EQ(get_accuracy(*host_models, 0), 17);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain->clone_to<Device>();
    auto device_models = host_models->clone_to<Device>(device_domain);
    auto device_candidates = host_candidates->clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(*device_models, 0), 13);
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu(random, device_models, device_candidates)
      .improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 19);
  }
}

TEST(ImproveProfiles, Larger) {
  std::mt19937 gen(42);
  auto model = generate::model(&gen, 4, 5);

  std::fill(model.weights.begin(), model.weights.end(), 0.5);

  auto learning_set = generate::learning_set(&gen, model, 250);
  auto host_domain = Domain<Host>::make(learning_set);
  auto host_candidates = Candidates<Host>::make(host_domain);

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
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnCpu(random, host_candidates)
      .improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*host_models, 0), 162);
  }

  {
    auto host_models = make_host_models();
    auto device_domain = host_domain->clone_to<Device>();
    auto device_models = host_models->clone_to<Device>(device_domain);
    auto device_candidates = host_candidates->clone_to<Device>(device_domain);

    EXPECT_EQ(get_accuracy(*device_models, 0), 132);
    ImproveProfilesWithAccuracyHeuristicWithMidpointCandidatesOnGpu(random, device_models, device_candidates)
      .improve_profiles(host_models);
    EXPECT_EQ(get_accuracy(*device_models, 0), 157);
  }
}

}  // namespace ppl
