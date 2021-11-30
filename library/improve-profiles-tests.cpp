// Copyright 2021 Vincent Jacques

#include <gtest/gtest.h>

#include "assign.hpp"
#include "generate.hpp"
#include "improve-profiles.hpp"


using ppl::Domain;
using ppl::Models;
using ppl::improve_profiles::Desirability;
using ppl::get_accuracy;
using ppl::improve_profiles::improve_profiles;

namespace ppl::improve_profiles {

// Internal functions (not declared in the header) that we still want to unit-test

__host__ __device__ Desirability compute_move_desirability(
  const ModelsView&,
  uint model_index,
  uint profile_index,
  uint criterion_index,
  float destination);

}  // namespace ppl::improve_profiles

Domain<Host> make_domain(
  const uint categories_count,
  const std::vector<std::pair<std::vector<float>, uint>>& alternatives_
) {
  const uint criteria_count = alternatives_.front().first.size();
  const uint alternatives_count = alternatives_.size();

  std::vector<ppl::io::ClassifiedAlternative> alternatives;
  for (auto alternative : alternatives_) {
    alternatives.push_back(ppl::io::ClassifiedAlternative(alternative.first, alternative.second));
  }

  return Domain<Host>::make(ppl::io::LearningSet(criteria_count, categories_count, alternatives_count, alternatives));
}

Models<Host> make_models(
  const Domain<Host>& domain,
  const std::vector<std::pair<std::vector<std::vector<float>>, std::vector<float>>>& models_
) {
  const uint criteria_count = domain.get_view().criteria_count;
  const uint categories_count = domain.get_view().categories_count;
  const uint models_count = models_.size();

  std::vector<ppl::io::Model> models;
  for (auto model : models_) {
    models.push_back(ppl::io::Model(criteria_count, categories_count, model.first, model.second));
  }

  return Models<Host>::make(domain, models);
}

Desirability compute_move_desirability(
    const Models<Host>& models,
    uint model_index,
    uint profile_index,
    uint criterion_index,
    float destination) {
  return ppl::improve_profiles::compute_move_desirability(
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
  auto domain = make_domain(2, {{{0.5}, 0}});
  auto models = make_models(domain, {{{{0.1}}, {1}}});

  auto device_domain = domain.clone_to<Device>();
  auto device_models = models.clone_to<Device>(device_domain);

  RandomSource random;
  random.init_for_host(42);
  random.init_for_device(42);

  EXPECT_EQ(get_accuracy(models, 0), 0);
  improve_profiles(random, &models);
  EXPECT_EQ(get_accuracy(models, 0), 1);

  EXPECT_EQ(get_accuracy(device_models, 0), 0);
  improve_profiles(random, &device_models);
  EXPECT_EQ(get_accuracy(device_models, 0), 1);
}
