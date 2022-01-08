// Copyright 2021-2022 Vincent Jacques

#include <ortools/glop/lp_solver.h>
#include <ortools/lp_data/lp_data.h>
#include <ortools/lp_data/lp_types.h>

#include <memory>

#include <chrones.hpp>

#include "../assign.hpp"
#include "../generate.hpp"
#include "../test-utils.hpp"
#include "glop.hpp"


CHRONABLE("glop-tests")

namespace ppl {

// Internal function (not declared in the header) that we still want to unit-test
std::shared_ptr<operations_research::glop::LinearProgram> make_verbose_linear_program(
  const float epsilon, std::shared_ptr<Models<Host>>, uint model_index);


TEST(GlopExploration, FromSample) {
  // Exploration test inspired from
  // https://github.com/google/or-tools/blob/stable/ortools/glop/samples/simple_glop_program.cc
  // and simplified

  operations_research::glop::LinearProgram lp;

  operations_research::glop::ColIndex col_x = lp.CreateNewVariable();
  lp.SetVariableBounds(col_x, -30.0, 6.0);
  operations_research::glop::ColIndex col_y = lp.CreateNewVariable();
  lp.SetVariableBounds(col_y, -3.0, 60.0);

  operations_research::glop::RowIndex row_r1 = lp.CreateNewConstraint();
  lp.SetConstraintBounds(row_r1, 3.0, 3.0);
  lp.SetCoefficient(row_r1, col_x, 1);
  lp.SetCoefficient(row_r1, col_y, 1);

  lp.SetObjectiveCoefficient(col_x, 2);
  lp.SetObjectiveCoefficient(col_y, 1);
  lp.SetMaximizationProblem(true);

  EXPECT_EQ(lp.num_variables(), 2);
  EXPECT_EQ(lp.num_constraints(), 1);
  EXPECT_EQ(
    lp.Dump(),
    "max: + 2 c0 + c1;\n"
    "r0: + c0 + c1 = 3;\n"
    "-30 <= c0 <= 6;\n"
    "-3 <= c1 <= 60;\n");

  operations_research::glop::LPSolver solver;
  operations_research::glop::GlopParameters parameters;
  parameters.set_provide_strong_optimal_guarantee(true);
  solver.SetParameters(parameters);

  EXPECT_EQ(solver.Solve(lp), operations_research::glop::ProblemStatus::OPTIMAL);

  EXPECT_NEAR(solver.GetObjectiveValue(), 9, 1e-6);

  const operations_research::glop::DenseRow& values = solver.variable_values();
  EXPECT_NEAR(values[col_x], 6, 1e-6);
  EXPECT_NEAR(values[col_y], -3, 1e-6);
}


TEST(MakeLinearProgram, OneCriterionOneAlternativeBelowProfileInBottomCategory) {
  auto domain = make_domain(2, {
    {{0}, 0},
  });
  auto models = make_models(domain, {
    {{{0.5}}, {1}},
  });

  EXPECT_EQ(
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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
    make_verbose_linear_program(0.125, models, 0)->Dump(),
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

TEST(OptimizeWeightsUsingGlop, First) {
  auto domain = make_domain(2, {{{1}, 1}});
  auto models = make_models(domain, {{{{0.5}}, {0.1}}});

  EXPECT_EQ(get_accuracy(*models, 0), 0);
  OptimizeWeightsUsingGlop().optimize_weights(models);
  EXPECT_EQ(get_accuracy(*models, 0), 1);
}

TEST(OptimizeWeightsUsingGlop, Larger) {
  std::mt19937 gen(57);
  auto model = generate::model(&gen, 4, 3);
  auto learning_set = generate::learning_set(&gen, model, 1000);

  std::fill(model.weights.begin(), model.weights.end(), 0.25);

  auto domain = Domain<Host>::make(learning_set);
  auto models = Models<Host>::make(domain, {model});

  EXPECT_EQ(get_accuracy(*models, 0), 233);
  OptimizeWeightsUsingGlop().optimize_weights(models);
  EXPECT_EQ(get_accuracy(*models, 0), 1000);
}

}  // namespace ppl
