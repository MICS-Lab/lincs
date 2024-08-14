// Copyright 2024 Vincent Jacques

#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <variant>

#include <boost/format.hpp>

#include "custom-on-cpu.hpp"
#include "../vendored/lov-e.hpp"

#include "../vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace lincs {

namespace {
  constexpr float infinity = std::numeric_limits<float>::infinity();

  bool verbose = false;
}

CustomOnCpuVerbose::CustomOnCpuVerbose() {
  verbose = true;
}

CustomOnCpuVerbose::~CustomOnCpuVerbose() {
  verbose = false;
}


struct Tableau {
  const unsigned client_variables_count;
  const unsigned slack_variables_count;
  const unsigned artificial_variables_count;
  Array2D<Host, float> objectives_coefficients;
  Array1D<Host, float> costs;
  Array2D<Host, float> constraints_coefficients;
  Array1D<Host, float> constraints_values;
  std::vector<unsigned> basic_variable_cols;

  std::vector<float> get_assignments() const {
    std::vector<float> assignments(client_variables_count + slack_variables_count + artificial_variables_count, 0);
    for (unsigned basic_variable_row = 0; basic_variable_row < constraints_values.s0(); ++basic_variable_row) {
      const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
      assignments[basic_variable_col] = constraints_values[basic_variable_row];
    }
    return assignments;
  }
};


class Simplex {
 public:
  Simplex(Tableau& tableau) :
    client_variables_count(tableau.client_variables_count),
    slack_variables_count(tableau.slack_variables_count),
    artificial_variables_count(tableau.artificial_variables_count),
    total_variables_count(client_variables_count + slack_variables_count + artificial_variables_count),
    objectives_count(tableau.objectives_coefficients.s1()),
    objectives_coefficients(tableau.objectives_coefficients),
    costs(tableau.costs),
    constraints_count(tableau.constraints_coefficients.s1()),
    constraints_coefficients(tableau.constraints_coefficients),
    constraints_values(tableau.constraints_values),
    basic_variable_cols(tableau.basic_variable_cols)
  {
    print_state("Initial");
    assert_invariants();
  }

 public:
  struct Optimal {};
  struct Unbounded {};
  typedef std::variant<Optimal, Unbounded> RunResult;

 private:
  struct Pivot {
    unsigned row;
    unsigned column;
  };
  typedef std::variant<Optimal, Unbounded, Pivot> StepResult;

 public:
  RunResult run() {
    while (true) {
      const auto step_result = step();
      if (std::holds_alternative<Simplex::Optimal>(step_result)) {
        return Optimal{};
      } else if (std::holds_alternative<Simplex::Unbounded>(step_result)) {
        return Unbounded{};
      } else {
        std::get<Simplex::Pivot>(step_result);
      }
    }
  }

 private:
  StepResult step() {
    std::optional<unsigned> entering_column = find_entering_column();
    if (entering_column) {
      const std::string entering = variable_name(*entering_column);
      std::optional<unsigned> leaving_row = find_leaving_row(*entering_column);
      if (leaving_row) {
        const std::string leaving = variable_name(basic_variable_cols[*leaving_row]);
        pivot(*leaving_row, *entering_column);
        print_state(boost::format("After pivot (entering column: %||, leaving row: %||)") % entering % leaving);
        assert_invariants();
        return Pivot{*leaving_row, *entering_column};
      } else {
        if (verbose) {
          std::cout << boost::format("  Unbounded! (entering column: %||)") % entering << std::endl;
        }
        return Unbounded{};
      }
    } else {
      if (verbose) {
        std::cout << "  Optimal!" << std::endl;
      }
      return Optimal{};
    }
  }

  std::optional<unsigned> find_entering_column() const {
    const auto& objective_coefficients = objectives_coefficients[objectives_count - 1];

    unsigned entering_column = 0;
    for (unsigned col = 1; col != total_variables_count; ++col) {
      if (objective_coefficients[col] > objective_coefficients[entering_column]) {
        entering_column = col;
      }
    }

    if (objective_coefficients[entering_column] > 0) {
      return entering_column;
    } else {
      return {};
    }
  }

  std::optional<unsigned> find_leaving_row(const unsigned entering_column) const {
    unsigned leaving_row = 0;
    for (unsigned row = 1; row != constraints_count; ++row) {
      if (constraints_coefficients[row][entering_column] > 0) {
        if (constraints_coefficients[leaving_row][entering_column] <= 0 || constraints_values[row] / constraints_coefficients[row][entering_column] < constraints_values[leaving_row] / constraints_coefficients[leaving_row][entering_column]) {
          leaving_row = row;
        }
      }
    }

    if (constraints_coefficients[leaving_row][entering_column] > 0) {
      return leaving_row;
    } else {
      return {};
    }
  }

  void pivot(const unsigned leaving_row, const unsigned entering_column) {
    const float pivot_value = constraints_coefficients[leaving_row][entering_column];
    for (unsigned col = 0; col != total_variables_count; ++col) {
      constraints_coefficients[leaving_row][col] /= pivot_value;
    }
    constraints_values[leaving_row] /= pivot_value;

    for (unsigned row = 0; row != constraints_count; ++row) {
      if (row != leaving_row) {
        const float factor = constraints_coefficients[row][entering_column];
        for (unsigned col = 0; col != total_variables_count; ++col) {
          constraints_coefficients[row][col] -= factor * constraints_coefficients[leaving_row][col];
        }
        constraints_values[row] -= factor * constraints_values[leaving_row];
      }
    }
    for (unsigned row = 0; row != objectives_count; ++row) {
      const float factor = objectives_coefficients[row][entering_column];
      for (unsigned col = 0; col != total_variables_count; ++col) {
        objectives_coefficients[row][col] -= factor * constraints_coefficients[leaving_row][col];
      }
      costs[row] -= factor * constraints_values[leaving_row];
    }

    basic_variable_cols[leaving_row] = entering_column;
  }

 private:
  template <typename Header>
  void print_state(const Header& header) {
    if (verbose) {
      std::cout << "  " << header << ':' << "\n";

      std::cout << "    constraints:\n          |";
      for (unsigned col = 0; col < total_variables_count; ++col) {
        std::cout << boost::format("%|=8|") % variable_name(col);
      }
      std::cout << "|\n";
      for (unsigned row = 0; row < constraints_count; ++row) {
        std::cout << boost::format("      %|3| |") % variable_name(basic_variable_cols[row]);
        for (unsigned col = 0; col < total_variables_count; ++col) {
          std::cout << boost::format("%|=8.3|") % constraints_coefficients[row][col];
        }
        std::cout << boost::format("| %|-.3|\n") % constraints_values[row];
      }

      std::cout << "    objectives:\n";
      for (unsigned row = 0; row != objectives_count; ++row) {
        std::cout << "          |";
        for (unsigned col = 0; col < total_variables_count; ++col) {
          std::cout << boost::format("%|=8.3|") % objectives_coefficients[row][col];
        }
        std::cout << boost::format("| %|-.3|") % costs[row] << std::endl;
      }
    }
  }

  std::string variable_name(const unsigned variable_index) const {
    if (variable_index < client_variables_count) {
      return "x" + std::to_string(1 + variable_index);
    } else if (variable_index < client_variables_count + slack_variables_count) {
      return "z" + std::to_string(1 + variable_index - client_variables_count);
    } else {
      return "y" + std::to_string(1 + variable_index - client_variables_count - slack_variables_count);
    }
  }

 private:
  void assert_invariants() {
    assert_sizes_are_consistent();
    assert_basic_variables_are_consistent();
  }

  void assert_sizes_are_consistent() {
    assert(costs.s0() == objectives_count);
    assert(objectives_coefficients.s1() == objectives_count);
    assert(objectives_coefficients.s0() == total_variables_count);
    assert(constraints_coefficients.s1() == constraints_count);
    assert(constraints_coefficients.s0() == total_variables_count);
    assert(constraints_values.s0() == constraints_count);
    assert(basic_variable_cols.size() == constraints_count);
  }

  void assert_basic_variables_are_consistent() {
    for (unsigned basic_variable_row = 0; basic_variable_row != constraints_count; ++basic_variable_row) {
      const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
      for (unsigned row = 1; row < constraints_count; ++row) {
        assert(constraints_coefficients[row][basic_variable_col] == (row == basic_variable_row ? 1. : 0.));
      }
    }
  }

 private:
  const unsigned client_variables_count;
  const unsigned slack_variables_count;
  const unsigned artificial_variables_count;
  const unsigned total_variables_count;
  const unsigned objectives_count;
  Array2D<Host, float>& objectives_coefficients;
  Array1D<Host, float>& costs;
  const unsigned constraints_count;
  Array2D<Host, float>& constraints_coefficients;
  Array1D<Host, float>& constraints_values;
  std::vector<unsigned>& basic_variable_cols;
};

class CustomOnCpuLinearProgramSolver {
 public:
  CustomOnCpuLinearProgramSolver(const CustomOnCpuLinearProgram& program) :
    program(program)
  {
    print_program();
  }

 private:
  void print_program() const {
    if (verbose) {
      std::cout << "PROGRAM" << std::endl;
      std::cout << "=======" << std::endl;
      std::cout << "  Objective:\n    minimize ";
      print_coefficients(program.get_objective_coefficients());
      std::cout << std::endl;

      std::cout << "  Subject to:" << std::endl;
      for (const auto& constraint : program.get_constraints()) {
        if (constraint.upper_bound == constraint.lower_bound) {
            std::cout << "    ";
            print_coefficients(constraint.coefficients);
            std::cout << "== " << constraint.upper_bound << std::endl;
        } else {
          if (constraint.upper_bound != infinity) {
            std::cout << "    ";
            print_coefficients(constraint.coefficients);
            std::cout << "<= " << constraint.upper_bound << std::endl;
          }
          if (constraint.lower_bound != -infinity) {
            std::cout << "    ";
            print_coefficients(constraint.coefficients);
            std::cout << ">= " << constraint.lower_bound << std::endl;
          }
        }
      }
    }
  }

  static void print_coefficients(const std::map<CustomOnCpuLinearProgram::variable_type, float>& coefficients) {
    for (const auto& [variable, coefficient] : coefficients) {
      std::cout << std::showpos << coefficient << std::noshowpos << "*x" << variable + 1 << " ";
    }
  }

 public:
  CustomOnCpuLinearProgram::solution_type solve() {
    const auto [constraints_count, slack_variables_count, artificial_variables_count] = count_constraints_and_additional_variables();
    if (artificial_variables_count == 0) {
      if (verbose) {
        std::cout << "SINGLE PHASE" << std::endl;
        std::cout << "============" << std::endl;
      }
      auto tableau = make_single_step_tableau(constraints_count, slack_variables_count);
      const auto run_result = Simplex(tableau).run();
      if (verbose) {
        std::cout << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(run_result)) {
        return {tableau.get_assignments(), tableau.costs[0]};
      } else {
        return {tableau.get_assignments(), -infinity};
      }
    } else {
      if (verbose) {
        std::cout << "FIRST PHASE" << std::endl;
        std::cout << "===========" << std::endl;
      }
      auto first_tableau = make_first_step_tableau(constraints_count, slack_variables_count, artificial_variables_count);
      Simplex(first_tableau).run();  // We don't care if it's optimal or unbounded

      const auto first_assignments = first_tableau.get_assignments();
      for (unsigned artificial_variable_index = first_tableau.client_variables_count + first_tableau.slack_variables_count; artificial_variable_index < first_tableau.client_variables_count + first_tableau.slack_variables_count + first_tableau.artificial_variables_count; ++artificial_variable_index) {
        if (first_assignments[artificial_variable_index] != 0) {
          if (verbose) {
            std::cout << "  Infeasible!" << std::endl;
          }
          return {first_assignments, std::numeric_limits<float>::quiet_NaN()};
        }
      }

      if (verbose) {
        std::cout << "SECOND PHASE" << std::endl;
        std::cout << "============" << std::endl;
      }

      auto second_tableau = make_second_step_tableau(first_tableau);
      const auto second_run_result = Simplex(second_tableau).run();
      if (verbose) {
        std::cout << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(second_run_result)) {
        return {second_tableau.get_assignments(), second_tableau.costs[0]};
      } else {
        return {second_tableau.get_assignments(), -infinity};
      }
    }
  }

 private:
  Tableau make_single_step_tableau(const unsigned constraints_count, const unsigned slack_variables_count) {
    const unsigned client_variables_count = program.variables_count();
    const unsigned artificial_variables_count = 0;
    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;

    Array2D<Host, float> constraints_coefficients(constraints_count, total_variables_count, zeroed);
    Array1D<Host, float> constraints_values(constraints_count, zeroed);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      constraints_coefficients,
      constraints_values,
      basic_variable_cols,
      artificial_variable_rows
    );

    Array2D<Host, float> objectives_coefficients(1, total_variables_count, zeroed);
    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      objectives_coefficients[0][variable_index] = -coefficient;
    }
    Array1D<Host, float> costs(1, zeroed);

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      std::move(objectives_coefficients),
      std::move(costs),
      std::move(constraints_coefficients),
      std::move(constraints_values),
      std::move(basic_variable_cols)
    };
  }

  Tableau make_first_step_tableau(const unsigned constraints_count, const unsigned slack_variables_count, const unsigned artificial_variables_count) {
    const unsigned client_variables_count = program.variables_count();
    assert(artificial_variables_count != 0);
    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;

    Array2D<Host, float> constraints_coefficients(constraints_count, total_variables_count, zeroed);
    Array1D<Host, float> constraints_values(constraints_count, zeroed);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      constraints_coefficients,
      constraints_values,
      basic_variable_cols,
      artificial_variable_rows
    );

    Array2D<Host, float> objectives_coefficients(2, total_variables_count, zeroed);
    Array1D<Host, float> costs(2, zeroed);
    // Phase 2 objective
    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      objectives_coefficients[0][variable_index] = -coefficient;
    }
    // Phase 1 objective...
    for (unsigned artificial_variable_index = client_variables_count + slack_variables_count; artificial_variable_index < total_variables_count; ++artificial_variable_index) {
      objectives_coefficients[1][artificial_variable_index] = -1;
    }
    // ... in term of non-basic variables
    for (unsigned artificial_variable_index = 0; artificial_variable_index != artificial_variables_count; ++artificial_variable_index) {
      const unsigned row = artificial_variable_rows[artificial_variable_index];
      for (unsigned col = 0; col < total_variables_count; ++col) {
        objectives_coefficients[1][col] += constraints_coefficients[row][col];
      }
      costs[1] += constraints_values[row];
    }

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      std::move(objectives_coefficients),
      std::move(costs),
      std::move(constraints_coefficients),
      std::move(constraints_values),
      std::move(basic_variable_cols)
    };
  }

  std::tuple<unsigned, unsigned, unsigned> count_constraints_and_additional_variables() const {
    unsigned slack_variables_count = 0;
    unsigned artificial_variables_count = 0;
    unsigned constraints_count = 0;
    for (const auto& constraint : program.get_constraints()) {
      if (constraint.upper_bound == constraint.lower_bound) {
        ++artificial_variables_count;
        ++constraints_count;
      } else {
        if (constraint.upper_bound != infinity) {
          ++slack_variables_count;
          if (constraint.upper_bound < 0) {
            ++artificial_variables_count;
          }
          ++constraints_count;
        }
        if (constraint.lower_bound != -infinity) {
          ++slack_variables_count;
          if (constraint.lower_bound > 0) {
            ++artificial_variables_count;
          }
          ++constraints_count;
        }
      }
    }
    return {constraints_count, slack_variables_count, artificial_variables_count};
  }

  void populate_constraints(
    const unsigned constraints_count,
    const unsigned client_variables_count,
    const unsigned slack_variables_count,
    const unsigned artificial_variables_count,
    Array2D<Host, float>& constraints_coefficients,
    Array1D<Host, float>& constraints_values,
    std::vector<unsigned>& basic_variable_cols,
    std::vector<unsigned>& artificial_variable_rows
  ) {
    unsigned slack_variable_index = 0;
    unsigned artificial_variable_index = 0;
    unsigned constraint_index = 0;
    for (const auto& constraint : program.get_constraints()) {
      // @todo Factorize common parts between these branches. Maybe 'std::sign(...) * ...' can help us?
      if (constraint.upper_bound == constraint.lower_bound) {
        for (const auto& [variable_index, coefficient] : constraint.coefficients) {
          constraints_coefficients[constraint_index][variable_index] = coefficient;
        }
        constraints_values[constraint_index] = constraint.upper_bound;

        constraints_coefficients[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
        basic_variable_cols[constraint_index] = client_variables_count + slack_variables_count + artificial_variable_index;
        artificial_variable_rows[artificial_variable_index] = constraint_index;
        ++artificial_variable_index;

        ++constraint_index;
      } else {
        if (constraint.upper_bound != infinity) {
          if (constraint.upper_bound >= 0) {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              constraints_coefficients[constraint_index][variable_index] = coefficient;
            }
            constraints_values[constraint_index] = constraint.upper_bound;

            constraints_coefficients[constraint_index][client_variables_count + slack_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;
          } else {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              constraints_coefficients[constraint_index][variable_index] = -coefficient;
            }
            constraints_values[constraint_index] = -constraint.upper_bound;

            constraints_coefficients[constraint_index][client_variables_count + slack_variable_index] = -1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;

            constraints_coefficients[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variables_count + artificial_variable_index;
            artificial_variable_rows[artificial_variable_index] = constraint_index;
            ++artificial_variable_index;
          }

          ++constraint_index;
        }
        if (constraint.lower_bound != -infinity) {
          if (constraint.lower_bound <= 0) {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              constraints_coefficients[constraint_index][variable_index] = -coefficient;
            }
            constraints_values[constraint_index] = -constraint.lower_bound;

            constraints_coefficients[constraint_index][client_variables_count + slack_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;
          } else {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              constraints_coefficients[constraint_index][variable_index] = coefficient;
            }
            constraints_values[constraint_index] = constraint.lower_bound;

            constraints_coefficients[constraint_index][client_variables_count + slack_variable_index] = -1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;

            constraints_coefficients[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variables_count + artificial_variable_index;
            artificial_variable_rows[artificial_variable_index] = constraint_index;
            ++artificial_variable_index;
          }

          ++constraint_index;
        }
      }
    }
    assert(constraint_index == constraints_count);
    assert(slack_variable_index == slack_variables_count);
    assert(artificial_variable_index == artificial_variables_count);
  }

  Tableau make_second_step_tableau(const Tableau& first_tableau) {
    // @todo Replace these copies by a partial reuse of the first tableau. It's just a matter of indexing, right?

    const unsigned client_variables_count = first_tableau.client_variables_count;
    const unsigned slack_variables_count = first_tableau.slack_variables_count;
    const unsigned artificial_variables_count = 0;

    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;
    const unsigned constraints_count = first_tableau.constraints_coefficients.s1();

    Array2D<Host, float> objectives_coefficients(1, total_variables_count, zeroed);
    for (unsigned variable_index = 0; variable_index != total_variables_count; ++variable_index) {
      objectives_coefficients[0][variable_index] = first_tableau.objectives_coefficients[0][variable_index];
    }
    Array1D<Host, float> costs(1, zeroed);
    costs[0] = first_tableau.costs[0];

    Array2D<Host, float> constraints_coefficients(constraints_count, total_variables_count, zeroed);
    for (unsigned row = 0; row != constraints_count; ++row) {
      for (unsigned col = 0; col != total_variables_count; ++col) {
        constraints_coefficients[row][col] = first_tableau.constraints_coefficients[row][col];
      }
    }
    Array1D<Host, float> constraints_values(constraints_count, zeroed);
    for (unsigned row = 0; row != constraints_count; ++row) {
      constraints_values[row] = first_tableau.constraints_values[row];
    }
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    for (unsigned row = 0; row != constraints_count; ++row) {
      assert(first_tableau.basic_variable_cols[row] < total_variables_count);
      basic_variable_cols[row] = first_tableau.basic_variable_cols[row];
    }

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      std::move(objectives_coefficients),
      std::move(costs),
      std::move(constraints_coefficients),
      std::move(constraints_values),
      std::move(basic_variable_cols)
    };
  };

 private:
  const CustomOnCpuLinearProgram& program;
};

CustomOnCpuLinearProgram::solution_type CustomOnCpuLinearProgram::solve() {
  return CustomOnCpuLinearProgramSolver(*this).solve();
}

}  // namespace lincs
