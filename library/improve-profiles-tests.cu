// Copyright 2021 Vincent Jacques

#include "improve-profiles.hpp"

#include <gtest/gtest.h>

#include <utility>

#include "cuda-utils.hpp"


using ppl::improve_profiles::Domain;
using ppl::improve_profiles::Models;
using ppl::improve_profiles::Desirability;

Domain<Host> make_domain(
  const int categories_count,
  const std::vector<std::pair<std::vector<float>, int>>& alternatives_
) {
  const int criteria_count = alternatives_.front().first.size();
  const int alternatives_count = alternatives_.size();

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
  const int criteria_count = domain.criteria_count;
  const int categories_count = domain.categories_count;
  const int models_count = models_.size();

  std::vector<ppl::io::Model> models;
  for (auto model : models_) {
    models.push_back(ppl::io::Model(criteria_count, categories_count, model.first, model.second));
  }

  return Models<Host>::make(domain, models);
}

TEST(MakeModels, SingleAlternativeSingleCriteria) {
  auto domain = make_domain(2, {
    {{0.25}, 1},
  });

  EXPECT_EQ(domain.categories_count, 2);
  EXPECT_EQ(domain.criteria_count, 1);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_assignments[0], 1);

  {
    auto models = make_models(domain, {});
    EXPECT_EQ(models.models_count, 0);
  }

  {
    auto models = make_models(domain, {
      {{{0.25}}, {1.}},
    });

    EXPECT_EQ(models.models_count, 1);
    EXPECT_EQ(models.weights[0][0], 1.);
    EXPECT_EQ(models.profiles[0][0][0], 0.25);
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

  EXPECT_EQ(domain.categories_count, 4);
  EXPECT_EQ(domain.criteria_count, 1);
  EXPECT_EQ(domain.learning_alternatives_count, 5);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.00);
  EXPECT_EQ(domain.learning_alternatives[0][1], 0.25);
  EXPECT_EQ(domain.learning_alternatives[0][2], 0.50);
  EXPECT_EQ(domain.learning_alternatives[0][3], 0.75);
  EXPECT_EQ(domain.learning_alternatives[0][4], 1.00);
  EXPECT_EQ(domain.learning_assignments[0], 0);
  EXPECT_EQ(domain.learning_assignments[1], 1);
  EXPECT_EQ(domain.learning_assignments[2], 2);
  EXPECT_EQ(domain.learning_assignments[3], 2);
  EXPECT_EQ(domain.learning_assignments[4], 3);

  auto models = make_models(domain, {
    {{{0.25}, {0.50}, {0.75}}, {1.}},
  });

  EXPECT_EQ(models.models_count, 1);
  EXPECT_EQ(models.weights[0][0], 1.);
  EXPECT_EQ(models.profiles[0][0][0], 0.25);
  EXPECT_EQ(models.profiles[0][1][0], 0.50);
  EXPECT_EQ(models.profiles[0][2][0], 0.75);
}

TEST(MakeModels, SingleAlternativeSeveralCriteria) {
  auto domain = make_domain(2, {
    {{0.25, 0.75, 0.50}, 1},
  });

  EXPECT_EQ(domain.categories_count, 2);
  EXPECT_EQ(domain.criteria_count, 3);
  EXPECT_EQ(domain.learning_alternatives_count, 1);
  EXPECT_EQ(domain.learning_alternatives[0][0], 0.25);
  EXPECT_EQ(domain.learning_alternatives[1][0], 0.75);
  EXPECT_EQ(domain.learning_alternatives[2][0], 0.50);
  EXPECT_EQ(domain.learning_assignments[0], 1);

  auto models = make_models(domain, {
    {{{0.25, 0.50, 0.75}}, {0.25, 0.50, 0.25}},
  });

  EXPECT_EQ(models.models_count, 1);
  EXPECT_EQ(models.weights[0][0], 0.25);
  EXPECT_EQ(models.weights[1][0], 0.50);
  EXPECT_EQ(models.weights[2][0], 0.25);
  EXPECT_EQ(models.profiles[0][0][0], 0.25);
  EXPECT_EQ(models.profiles[1][0][0], 0.50);
  EXPECT_EQ(models.profiles[2][0][0], 0.75);
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

  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}}), 0, 0), 0);
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}}), 0, 1), 1);
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}}), 0, 2), 2);
  EXPECT_EQ(get_assignment(make_models(domain, {{{{0.3}, {0.5}, {0.7}}, {5}}}), 0, 3), 3);
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

  auto models = make_models(domain, {{{{0.9}}, {1}}, {{{0.5}}, {1}}});

  EXPECT_EQ(get_assignment(models, 0, 0), 0);
  EXPECT_EQ(get_assignment(models, 0, 1), 0);
  EXPECT_EQ(get_accuracy(models, 0), 1);
  EXPECT_EQ(get_assignment(models, 1, 0), 0);
  EXPECT_EQ(get_assignment(models, 1, 1), 1);
  EXPECT_EQ(get_accuracy(models, 1), 2);
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

  EXPECT_EQ(get_accuracy(models, 0), 0);

  improve_profiles(&models);

  EXPECT_EQ(get_accuracy(models, 0), 1);
}
