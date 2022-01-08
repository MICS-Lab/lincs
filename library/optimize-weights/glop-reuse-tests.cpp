// Copyright 2021-2022 Vincent Jacques

#include <ortools/glop/lp_solver.h>
#include <ortools/lp_data/lp_data.h>
#include <ortools/lp_data/lp_types.h>
#include <valgrind/valgrind.h>

#include <chrono>  // NOLINT(build/c++11)
#include <memory>

#include <chrones.hpp>

#include "../assign.hpp"
#include "../generate.hpp"
#include "../test-utils.hpp"
#include "glop-reuse.hpp"
#include "glop.hpp"


CHRONABLE("glop-reuse-tests")

namespace ppl {

namespace glp = operations_research::glop;

// Internal function (not declared in the header) that we still want to unit-test
std::shared_ptr<operations_research::glop::LinearProgram> make_verbose_linear_program_reuse(
  const float epsilon, std::shared_ptr<Models<Host>>, uint model_index);


TEST(MakeLinearProgram, OneCriterionOneAlternativeBelowProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{0}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5}}, {1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, OneCriterionOneAlternativeBelowProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{0}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5}}, {1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, OneCriterionOneAlternativeAboveProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{1}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5}}, {1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, OneCriterionOneAlternativeAboveProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{1}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5}}, {1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, TwoCriteriaOneAlternativeBelowProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{0, 0}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, TwoCriteriaOneAlternativeBelowProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{0, 0}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, TwoCriteriaOneAlternativeAboveProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{1, 1}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 + w_1 + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, TwoCriteriaOneAlternativeAboveProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{1, 1}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 + w_1 - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, TwoCriteriaOneAlternativeBelowThenAboveProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{0, 1}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_1 + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, TwoCriteriaOneAlternativeBelowThenAboveProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{0, 1}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_1 - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, TwoCriteriaOneAlternativeAboveThenBelowProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{1, 0}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 + y_0 - y'_0 = 0.875;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}

TEST(MakeLinearProgram, TwoCriteriaOneAlternativeAboveThenBelowProfileInTopCategory) {
  auto domain = make_domain(2, {
    {{1, 0}, 1},
  });
  auto models = make_models(domain, {
    {{{0.5, 0.5}}, {1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
    "min: + x'_0 + y'_0;\n"
    "r0: + w_0 - x_0 + x'_0 = 1;\n"
    "w_0 >= 0;\n"
    "w_1 >= 0;\n"
    "x_0 >= 0;\n"
    "x'_0 >= 0;\n"
    "y_0 >= 0;\n"
    "y'_0 >= 0;\n");
}


TEST(MakeLinearProgram, ThreeCriteriaAFewAlternatives) {
  auto domain = make_domain(3, {
    {{0, 0, 0}, 0},
    {{0.5, 0, 0}, 0},
    {{0.5, 0.5, 0}, 1},
    {{0.5, 0.5, 0.5}, 1},
    {{1, 0.5, 0.5}, 1},
    {{1, 1, 0.5}, 2},
    {{1, 1, 1}, 2},
  });
  auto models = make_models(domain, {
    {{{0.3, 0.3, 0.3}, {0.7, 0.7, 0.7}}, {1, 1, 1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program_reuse(0.125, models, 0)->Dump(),
      "min: + x'_0 + y'_0 + x'_1 + y'_1 + x'_2 + y'_2 + x'_3 + y'_3 + x'_4 + y'_4 + x'_5 + y'_5 + x'_6 + y'_6;\n"
      "r0: + y_0 - y'_0 = 0.875;\n"
      "r1: + w_0 + y_1 - y'_1 = 0.875;\n"
      "r2: + w_0 + w_1 - x_2 + x'_2 = 1;\n"
      "r3: + y_2 - y'_2 = 0.875;\n"
      "r4: + w_0 + w_1 + w_2 - x_3 + x'_3 = 1;\n"
      "r5: + y_3 - y'_3 = 0.875;\n"
      "r6: + w_0 + w_1 + w_2 - x_4 + x'_4 = 1;\n"
      "r7: + w_0 + y_4 - y'_4 = 0.875;\n"
      "r8: + w_0 + w_1 - x_5 + x'_5 = 1;\n"
      "r9: + w_0 + w_1 + w_2 - x_6 + x'_6 = 1;\n"
      "w_0 >= 0;\n"
      "w_1 >= 0;\n"
      "w_2 >= 0;\n"
      "x_0 >= 0;\n"
      "x'_0 >= 0;\n"
      "y_0 >= 0;\n"
      "y'_0 >= 0;\n"
      "x_1 >= 0;\n"
      "x'_1 >= 0;\n"
      "y_1 >= 0;\n"
      "y'_1 >= 0;\n"
      "x_2 >= 0;\n"
      "x'_2 >= 0;\n"
      "y_2 >= 0;\n"
      "y'_2 >= 0;\n"
      "x_3 >= 0;\n"
      "x'_3 >= 0;\n"
      "y_3 >= 0;\n"
      "y'_3 >= 0;\n"
      "x_4 >= 0;\n"
      "x'_4 >= 0;\n"
      "y_4 >= 0;\n"
      "y'_4 >= 0;\n"
      "x_5 >= 0;\n"
      "x'_5 >= 0;\n"
      "y_5 >= 0;\n"
      "y'_5 >= 0;\n"
      "x_6 >= 0;\n"
      "x'_6 >= 0;\n"
      "y_6 >= 0;\n"
      "y'_6 >= 0;\n");
}

TEST(OptimizeWeightsUsingGlopAndReusingPrograms, First) {
  auto domain = make_domain(2, {{{1}, 1}});
  auto models = make_models(domain, {{{{0.5}}, {0.1}}});

  EXPECT_EQ(get_accuracy(*models, 0), 0);
  OptimizeWeightsUsingGlopAndReusingPrograms(*models).optimize_weights(models);
  EXPECT_EQ(get_accuracy(*models, 0), 1);
}

TEST(OptimizeWeightsUsingGlopAndReusingPrograms, Larger) {
  std::mt19937 gen(57);
  auto model = generate::model(&gen, 4, 3);
  auto learning_set = generate::learning_set(&gen, model, 1000);

  std::fill(model.weights.begin(), model.weights.end(), 0.25);

  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, {model});

  EXPECT_EQ(get_accuracy(*models, 0), 233);
  OptimizeWeightsUsingGlopAndReusingPrograms(*models).optimize_weights(models);
  EXPECT_EQ(get_accuracy(*models, 0), 1000);
}

// Sadly in actual cases, models are modified too much to benefit from the optimization
TEST(OptimizeWeightsUsingGlopAndReusingPrograms, IsFasterThanOptimizeWeightsUsingGlopOnUnmodifiedModels) {
  if (RUNNING_ON_VALGRIND) return;

  std::mt19937 gen(57);
  auto model = generate::model(&gen, 6, 3);
  auto learning_set = generate::learning_set(&gen, model, 1000);

  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, std::vector<io::Model>(1, model));

  auto duration = std::chrono::seconds(2);

  int dont_reuse_count = 0;
  auto end = std::chrono::steady_clock::now() + duration;
  OptimizeWeightsUsingGlop dont_reuse;
  while (std::chrono::steady_clock::now() < end) {
    dont_reuse.optimize_weights(models);
    ++dont_reuse_count;
  }

  EXPECT_GE(dont_reuse_count, 5);

  int do_reuse_count = 0;
  end = std::chrono::steady_clock::now() + duration;
  OptimizeWeightsUsingGlopAndReusingPrograms do_reuse(*models);
  while (std::chrono::steady_clock::now() < end) {
    do_reuse.optimize_weights(models);
    ++do_reuse_count;
  }

  EXPECT_GT(do_reuse_count, 5 * dont_reuse_count);
}

}  // namespace ppl
