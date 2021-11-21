// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <gtest/gtest.h>

#include "generate.hpp"


using ppl::improve_profiles::Domain;
using ppl::improve_profiles::Models;
using ppl::improve_profiles::Desirability;

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
  return compute_move_desirability(models.get_view(), model_index, profile_index, criterion_index, destination);
}


TEST(MakeModels, SingleAlternativeSingleCriteria) {
  auto domain = make_domain(2, {
    {{0.25}, 1},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 2);
  EXPECT_EQ(domain_view.criteria_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain_view.learning_assignments[0], 1);

  {
    auto models = make_models(domain, {});
    auto models_view = models.get_view();

    EXPECT_EQ(models_view.models_count, 0);
    EXPECT_EQ(models_view.weights.s1(), 1);
    EXPECT_EQ(models_view.weights.s0(), 0);
    EXPECT_EQ(models_view.profiles.s2(), 1);
    EXPECT_EQ(models_view.profiles.s1(), 1);
    EXPECT_EQ(models_view.profiles.s0(), 0);
  }

  {
    auto models = make_models(domain, {
      {{{0.25}}, {1.}},
    });
    auto models_view = models.get_view();

    EXPECT_EQ(models_view.models_count, 1);
    EXPECT_EQ(models_view.weights.s1(), 1);
    EXPECT_EQ(models_view.weights.s0(), 1);
    EXPECT_EQ(models_view.weights[0][0], 1.);
    EXPECT_EQ(models_view.profiles.s2(), 1);
    EXPECT_EQ(models_view.profiles.s1(), 1);
    EXPECT_EQ(models_view.profiles.s0(), 1);
    EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  }
}

TEST(MakeModels, SeveralAlternativesSingleCriteria) {
  auto domain = make_domain(4, {
    {{0.00}, 0},
    {{0.25}, 1},
    {{0.50}, 2},
    {{0.75}, 2},
    {{1.00}, 3},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 4);
  EXPECT_EQ(domain_view.criteria_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives_count, 5);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.00);
  EXPECT_EQ(domain_view.learning_alternatives[0][1], 0.25);
  EXPECT_EQ(domain_view.learning_alternatives[0][2], 0.50);
  EXPECT_EQ(domain_view.learning_alternatives[0][3], 0.75);
  EXPECT_EQ(domain_view.learning_alternatives[0][4], 1.00);
  EXPECT_EQ(domain_view.learning_assignments[0], 0);
  EXPECT_EQ(domain_view.learning_assignments[1], 1);
  EXPECT_EQ(domain_view.learning_assignments[2], 2);
  EXPECT_EQ(domain_view.learning_assignments[3], 2);
  EXPECT_EQ(domain_view.learning_assignments[4], 3);

  auto models = make_models(domain, {
    {{{0.25}, {0.50}, {0.75}}, {1.}},
  });
  auto models_view = models.get_view();

  EXPECT_EQ(models_view.models_count, 1);
  EXPECT_EQ(models_view.weights[0][0], 1.);
  EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  EXPECT_EQ(models_view.profiles[0][1][0], 0.50);
  EXPECT_EQ(models_view.profiles[0][2][0], 0.75);
}

TEST(MakeModels, SingleAlternativeSeveralCriteria) {
  auto domain = make_domain(2, {
    {{0.25, 0.75, 0.50}, 1},
  });
  auto domain_view = domain.get_view();

  EXPECT_EQ(domain_view.categories_count, 2);
  EXPECT_EQ(domain_view.criteria_count, 3);
  EXPECT_EQ(domain_view.learning_alternatives_count, 1);
  EXPECT_EQ(domain_view.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain_view.learning_alternatives[1][0], 0.75);
  EXPECT_EQ(domain_view.learning_alternatives[2][0], 0.50);
  EXPECT_EQ(domain_view.learning_assignments[0], 1);

  auto models = make_models(domain, {
    {{{0.25, 0.50, 0.75}}, {0.25, 0.50, 0.25}},
  });
  auto models_view = models.get_view();

  EXPECT_EQ(models_view.models_count, 1);
  EXPECT_EQ(models_view.weights[0][0], 0.25);
  EXPECT_EQ(models_view.weights[1][0], 0.50);
  EXPECT_EQ(models_view.weights[2][0], 0.25);
  EXPECT_EQ(models_view.profiles[0][0][0], 0.25);
  EXPECT_EQ(models_view.profiles[1][0][0], 0.50);
  EXPECT_EQ(models_view.profiles[2][0][0], 0.75);
}

TEST(GetAssignment, SingleCriterion) {
  auto domain = make_domain(2, {{{0.5}, 0}});

  // Alternative above profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {5}}}), 0, 0), 1);
  // Alternative above profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {1}}}), 0, 0), 1);
  // Alternative above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.49}}, {0.99}}}), 0, 0), 0);

  // Alternative equal to profile, heavy weight => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {5}}}), 0, 0), 1);
  // Alternative equal to profile, weight just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {1}}}), 0, 0), 1);
  // Alternative equal to profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.5}}, {0.99}}}), 0, 0), 0);

  // Alternative below profile, whatever weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.51}}, {1}}}), 0, 0), 0);
}

TEST(GetAssignment, SingleCriterionManyCategories) {
  auto domain = make_domain(4, {{{0.2}, 0}, {{0.4}, 1}, {{0.6}, 2}, {{0.8}, 3}});
  auto models = make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}});

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 1);
  EXPECT_EQ(get_assignment(models, 0, 2), 2);
  EXPECT_EQ(get_assignment(models, 0, 3), 3);
}

TEST(GetAssignment, SeveralCriteria) {
  auto domain = make_domain(2, {{{0.3, 0.7}, 0}});

  // Alternative fully above profile, heavy weights => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {5, 5}}}), 0, 0), 1);
  // Alternative above profile on first criterion, heavy weight on first criterion => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.71}}, {5, 0.1}}}), 0, 0), 1);
  // Alternative above profile on second criterion, heavy weight on second criterion => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.31, 0.69}}, {0.1, 5}}}), 0, 0), 1);
  // Alternative fully above profile, weights just enough => reach C1
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {0.5, 0.5}}}), 0, 0), 1);
  // Alternative fully above profile, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.69}}, {0.49, 0.49}}}), 0, 0), 0);
  // Alternative above profile on first criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.29, 0.71}}, {0.99, 5}}}), 0, 0), 0);
  // Alternative above profile on second criterion, but insufficient weight => stay in C0
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.31, 0.69}}, {5, 0.99}}}), 0, 0), 0);
}

TEST(GetAssignmentAndAccuracy, SeveralAlternativesSeveralModels) {
  auto domain = make_domain(2, {{{0.25}, 0}, {{0.75}, 1}});
  auto device_domain = domain.clone_to<Device>();

  auto models = make_models(domain, {{{{0.9}}, {1}}, {{{0.5}}, {1}}});
  auto device_models = models.clone_to<Device>(device_domain);

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 0);
  EXPECT_EQ(get_accuracy(models, 0), 1);
  EXPECT_EQ(get_accuracy(device_models, 0), 1);
  EXPECT_EQ(get_assignment(models, 1, 0), 0);
  EXPECT_EQ(get_assignment(models, 1, 1), 1);
  EXPECT_EQ(get_accuracy(models, 1), 2);
  EXPECT_EQ(get_accuracy(device_models, 1), 2);
}

TEST(GetAccuracy, RandomDomainUniformModel) {
  std::mt19937 gen1(57);
  auto reference_model = ppl::generate::model(&gen1, 4, 5);
  std::fill(reference_model.weights.begin(), reference_model.weights.end(), 0.5);
  std::mt19937 gen2(57);
  auto learning_set = ppl::generate::learning_set(&gen2, reference_model, 100'000);

  auto model = ppl::io::Model::make_homogeneous(learning_set.criteria_count, 2., learning_set.categories_count);
  auto domain = ppl::improve_profiles::Domain<Host>::make(learning_set);
  auto models = ppl::improve_profiles::Models<Host>::make(domain, std::vector<ppl::io::Model>(1, model));

  const unsigned int expected_accuracy = 44'667;

  #ifndef NOSTOPWATCH
  for (int i = 0; i != 10; ++i)
  #endif
  ASSERT_EQ(get_accuracy(models, 0), expected_accuracy);

  auto device_domain = domain.clone_to<Device>();
  auto device_models = models.clone_to<Device>(device_domain);

  #ifndef NOSTOPWATCH
  for (int i = 0; i != 10; ++i)
  #endif
  ASSERT_EQ(get_accuracy(device_models, 0), expected_accuracy);
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
