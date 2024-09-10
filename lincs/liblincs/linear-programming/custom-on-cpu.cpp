// Copyright 2024 Vincent Jacques

#include <cassert>
#include <ctime>
#include <filesystem>
#include <fstream>
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


namespace fs = std::filesystem;

namespace {

constexpr float infinity = std::numeric_limits<float>::infinity();

std::string no_loss(const float f) {
  if (f == infinity) {
    return "infinity";
  } else if (f == -infinity) {
    return "-infinity";
  } else {
    for (unsigned precision = 3; precision != 20; ++precision) {
      std::ostringstream out;
      out << std::setprecision(precision) << f;
      std::istringstream in(out.str());
      float f2;
      in >> f2;
      if (f2 == f) {
        return out.str();
      }
    }
    assert(false);
  }
}

std::string generate(const lincs::CustomOnCpuLinearProgram& program, int verbose) {
  std::ostringstream out;
  out << "// Copyright 2023-2024 Vincent Jacques\n"
      << "\n"
      << "#include \"testing.hpp\"\n"
      << "\n"
      << "\n"
      << "TEST_CASE(\"Bug\") {\n"
      << "  test([](auto& linear_program) -> std::optional<float> {\n"
      << "    {\n";
  for (unsigned i = 0; i != program.variables_count(); ++i) {
    out << "      const auto x" << i << " = linear_program.create_variable();\n";
  }
  out << "      linear_program.mark_all_variables_created();\n";
  for (const auto& [variable, coefficient] : program.get_objective_coefficients()) {
    out << "      linear_program.set_objective_coefficient(x" << variable << ", " << no_loss(coefficient) << ");\n";
  }
  for (const auto& constraint : program.get_constraints()) {
    out << "      { "
      << "auto c = linear_program.create_constraint();"
      << " c.set_bounds(" << no_loss(constraint.lower_bound) << ", " << no_loss(constraint.upper_bound) << ");";
    for (const auto& [variable, coefficient] : constraint.coefficients) {
      out << " c.set_coefficient(x" << variable << ", " << no_loss(coefficient) << ");";
    }
    out << " }\n";
  }
  out << "    }\n";
  if (verbose == -1) {
    out << "    lincs::CustomOnCpuVerbose verbose;\n";
  } else if (verbose > 0) {
    out << "    lincs::CustomOnCpuVerbose verbose(" << verbose << ");\n";
  }
  out << "    const auto solution = linear_program.solve();\n"
      << "    if (solution) {\n"
      << "      return solution->cost;\n"
      << "    } else {\n"
      << "      return std::nullopt;\n"
      << "    }\n"
      << "  });\n"
      << "}\n";
  return out.str();
}

void dump(const lincs::CustomOnCpuLinearProgram& program) {
  std::string verbose = generate(program, -1);
  std::string verbose_1 = generate(program, 1);
  std::string verbose_2 = generate(program, 2);
  std::string verbose_3 = generate(program, 3);
  std::string quiet = generate(program, 0);

  bool found = false;
  for (const auto & entry : fs::directory_iterator("lincs/liblincs/linear-programming")) {
    std::ifstream file(entry.path());
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    if (content == verbose || content == verbose_1 || content == verbose_2 || content == verbose_3 || content == quiet) {
      found = true;
      break;
    }
  }
  if (!found) {
    std::ofstream file(str(boost::format("lincs/liblincs/linear-programming/test-bug-%||.cpp") % std::time(nullptr)));
    file << verbose_1;
  }
}

#ifdef NDEBUG
  #define assert_with_dump(expr, program) static_cast<void>(0)
#else
  #define assert_with_dump(expr, program) \
     (static_cast<bool>(expr) \
      ? static_cast<void>(0) \
      : (dump(program), __assert_fail(#expr, __FILE__, __LINE__, __func__)))
#endif

}  // namespace

namespace lincs {

namespace {

int verbosity = 0;

}  // namespace

CustomOnCpuVerbose::CustomOnCpuVerbose(const int v) {
  verbosity = v;
}

CustomOnCpuVerbose::~CustomOnCpuVerbose() {
  verbosity = 0;
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
        print_state(boost::format("After pivot (leaving row: %||, entering column: %||)") % leaving % entering);
        assert_invariants();
        return Pivot{*leaving_row, *entering_column};
      } else {
        if (verbosity > 1) {
          std::cerr << boost::format("  Unbounded! (entering column: %||)") % entering << std::endl;
        }
        return Unbounded{};
      }
    } else {
      if (verbosity > 1) {
        std::cerr << "  Optimal!" << std::endl;
      }
      return Optimal{};
    }
  }

  std::optional<unsigned> find_entering_column() const {
    const auto& objective_coefficients = objectives_coefficients[objectives_count - 1];

    std::optional<unsigned> entering_column;
    for (unsigned col = 0; col != total_variables_count; ++col) {
      if (objective_coefficients[col] > 0) {
        if (!entering_column || objective_coefficients[col] > objective_coefficients[*entering_column]) {
          entering_column = col;
        }
      }
    }

    return entering_column;
  }

  std::optional<unsigned> find_leaving_row(const unsigned entering_column) const {
    std::optional<unsigned> leaving_row;
    for (unsigned row = 0; row != constraints_count; ++row) {
      if (constraints_coefficients[row][entering_column] > 0) {
        if (!leaving_row || constraints_values[row] / constraints_coefficients[row][entering_column] < constraints_values[*leaving_row] / constraints_coefficients[*leaving_row][entering_column]) {
          leaving_row = row;
        }
      }
    }

    return leaving_row;
  }

  void pivot(const unsigned leaving_row, const unsigned entering_column) {
    if (verbosity > 2) {
      std::cerr << boost::format("  Pivoting (leaving row: %||, entering column: %||)") % variable_name(basic_variable_cols[leaving_row]) % variable_name(entering_column) << std::endl;
    }
    const float pivot_value = constraints_coefficients[leaving_row][entering_column];
    if (verbosity > 2) {
      std::cerr << boost::format("    pivot value: %|-.3|") % pivot_value << std::endl;
    }
    assert(pivot_value > 0);
    for (unsigned col = 0; col != total_variables_count; ++col) {
      constraints_coefficients[leaving_row][col] /= pivot_value;
    }
    assert(constraints_coefficients[leaving_row][entering_column] == 1);
    constraints_values[leaving_row] /= pivot_value;

    for (unsigned row = 0; row != constraints_count; ++row) {
      if (row != leaving_row) {
        const float factor = constraints_coefficients[row][entering_column];
        if (verbosity > 2) {
          std::cerr << boost::format("    constraint %|| factor: %|-.3|") % variable_name(basic_variable_cols[row]) % factor << std::endl;
        }
        for (unsigned col = 0; col != total_variables_count; ++col) {
          constraints_coefficients[row][col] -= factor * constraints_coefficients[leaving_row][col];
        }
        constraints_values[row] -= factor * constraints_values[leaving_row];
      }
    }
    for (unsigned row = 0; row != objectives_count; ++row) {
      const float factor = objectives_coefficients[row][entering_column];
      if (verbosity > 2) {
        std::cerr << boost::format("    objective %|| factor: %|-.3|") % row % factor << std::endl;
      }
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
    if (verbosity > 0) {
      std::cerr << "  " << header << ':' << "\n";

      if (verbosity > 1) {
        std::cerr << "    constraints:\n          |";
        for (unsigned col = 0; col < total_variables_count; ++col) {
          std::cerr << boost::format("%|=8|") % variable_name(col);
        }
        std::cerr << "|\n";
        for (unsigned row = 0; row < constraints_count; ++row) {
          std::cerr << boost::format("      %|3| |") % variable_name(basic_variable_cols[row]);
          for (unsigned col = 0; col < total_variables_count; ++col) {
            std::cerr << boost::format("%|=8.3|") % constraints_coefficients[row][col];
          }
          std::cerr << boost::format("| %|-.3|\n") % constraints_values[row];
        }

        std::cerr << "    objectives:\n";
        for (unsigned row = 0; row != objectives_count; ++row) {
          std::cerr << "          |";
          for (unsigned col = 0; col < total_variables_count; ++col) {
            std::cerr << boost::format("%|=8.3|") % objectives_coefficients[row][col];
          }
          std::cerr << boost::format("| %|-.3|") % costs[row] << std::endl;
        }
      } else {
        std::cerr << "    costs:";
        for (unsigned row = 0; row != objectives_count; ++row) {
          std::cerr << " " << costs[row];
        }
        std::cerr << std::endl;
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
    if (verbosity > 1) {
      std::cerr << "PROGRAM" << std::endl;
      std::cerr << "=======" << std::endl;
      std::cerr << "  Objective:\n    minimize ";
      print_coefficients(program.get_objective_coefficients());
      std::cerr << std::endl;

      std::cerr << "  Subject to:" << std::endl;
      for (const auto& constraint : program.get_constraints()) {
        if (constraint.upper_bound == constraint.lower_bound) {
            std::cerr << "    ";
            print_coefficients(constraint.coefficients);
            std::cerr << "== " << constraint.upper_bound << std::endl;
        } else {
          if (constraint.upper_bound != infinity) {
            std::cerr << "    ";
            print_coefficients(constraint.coefficients);
            std::cerr << "<= " << constraint.upper_bound << std::endl;
          }
          if (constraint.lower_bound != -infinity) {
            std::cerr << "    ";
            print_coefficients(constraint.coefficients);
            std::cerr << ">= " << constraint.lower_bound << std::endl;
          }
        }
      }
    }
  }

  static void print_coefficients(const std::map<CustomOnCpuLinearProgram::variable_type, float>& coefficients) {
    for (const auto& [variable, coefficient] : coefficients) {
      std::cerr << std::showpos << coefficient << std::noshowpos << "*x" << variable + 1 << " ";
    }
  }

 public:
  std::optional<CustomOnCpuLinearProgram::solution_type> solve() {
    const auto [constraints_count, slack_variables_count, artificial_variables_count] = count_constraints_and_additional_variables();
    if (artificial_variables_count == 0) {
      if (verbosity > 1) {
        std::cerr << "SINGLE PHASE" << std::endl;
        std::cerr << "============" << std::endl;
      }
      auto tableau = make_single_step_tableau(constraints_count, slack_variables_count);
      const auto run_result = Simplex(tableau).run();
      if (verbosity > 1) {
        std::cerr << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(run_result)) {
        if (verbosity > 0) {
          std::cerr << "OPTIMAL (single-phase)" << std::endl;
        }
        return CustomOnCpuLinearProgram::solution_type{tableau.get_assignments(), tableau.costs[0]};
      } else {
        if (verbosity > 0) {
          std::cerr << "UNBOUNDED (single-phase)" << std::endl;
        }
        return {};
      }
    } else {
      if (verbosity > 1) {
        std::cerr << "FIRST PHASE" << std::endl;
        std::cerr << "===========" << std::endl;
      }
      auto first_tableau = make_first_step_tableau(constraints_count, slack_variables_count, artificial_variables_count);
      Simplex(first_tableau).run();  // We don't care if it's optimal or unbounded

      const auto first_assignments = first_tableau.get_assignments();
      for (unsigned artificial_variable_index = 0; artificial_variable_index < first_tableau.artificial_variables_count; ++artificial_variable_index) {
        if (first_assignments[first_tableau.client_variables_count + first_tableau.slack_variables_count + artificial_variable_index] != 0) {
          if (verbosity > 1) {
            std::cerr << "  Infeasible!" << std::endl;
          }
          if (verbosity > 0) {
            std::cerr << "INFEASIBLE" << std::endl;
          }
          return {};
        }
      }
      if (verbosity > 0) {
        std::cerr << "FEASIBLE. First phase cost (should be zero): " << first_tableau.costs[1] << std::endl;
      }
      assert_with_dump(std::abs(first_tableau.costs[1]) < 1e-4, program);

      if (verbosity > 1) {
        std::cerr << "SECOND PHASE" << std::endl;
        std::cerr << "============" << std::endl;
      }

      auto second_tableau = make_second_step_tableau(first_tableau);
      const auto second_run_result = Simplex(second_tableau).run();
      if (verbosity > 1) {
        std::cerr << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(second_run_result)) {
        if (verbosity > 0) {
          std::cerr << "OPTIMAL (two phases)" << std::endl;
        }
        return CustomOnCpuLinearProgram::solution_type{second_tableau.get_assignments(), second_tableau.costs[0]};
      } else {
        if (verbosity > 0) {
          std::cerr << "UNBOUNDED (two phases)" << std::endl;
        }
        return {};
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
      assert_with_dump(first_tableau.basic_variable_cols[row] < total_variables_count, program);
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

std::optional<CustomOnCpuLinearProgram::solution_type> CustomOnCpuLinearProgram::solve() {
  return CustomOnCpuLinearProgramSolver(*this).solve();
}

}  // namespace lincs
