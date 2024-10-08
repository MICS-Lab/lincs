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
      << "    {\n"
      << "      std::vector<typename std::remove_reference_t<decltype(linear_program)>::variable_type> v;\n";
  for (unsigned i = 0; i != program.variables_count(); ++i) {
    out << "      v.push_back(linear_program.create_variable());\n";
  }
  out << "      linear_program.mark_all_variables_created();\n";
  for (const auto& [variable, coefficient] : program.get_objective_coefficients()) {
    out << "      linear_program.set_objective_coefficient(v[" << variable << "], " << no_loss(coefficient) << ");\n";
  }
  for (const auto& constraint : program.get_constraints()) {
    out << "      linear_program.create_constraint()"
      << ".set_bounds(" << no_loss(constraint.lower_bound) << ", " << no_loss(constraint.upper_bound) << ")";
    for (const auto& [variable, coefficient] : constraint.coefficients) {
      out << ".set_coefficient(v[" << variable << "], " << no_loss(coefficient) << ")";
    }
    out << ";\n";
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
  const unsigned total_variables_count;  // client_variables_count + slack_variables_count + artificial_variables_count
  const unsigned tableau_width;  // total_variables_count + 1
  const unsigned constraints_count;
  const unsigned objectives_count;
  const unsigned tableau_height;  // constraints_count + objectives_count
  typedef double fp_type;
  Array2D<Host, fp_type> tableau;
  std::vector<unsigned> basic_variable_cols;

  std::vector<float> get_assignments() const {
    std::vector<float> assignments(total_variables_count, 0);
    for (unsigned basic_variable_row = 0; basic_variable_row < constraints_count; ++basic_variable_row) {
      const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
      assignments[basic_variable_col] = tableau[basic_variable_row][total_variables_count];
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
    total_variables_count(tableau.total_variables_count),
    tableau_width(tableau.tableau_width),
    constraints_count(tableau.constraints_count),
    objectives_count(tableau.objectives_count),
    tableau_height(tableau.tableau_height),
    tableau(tableau.tableau),
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
      if (artificial_variables_count != 0) {
        bool artificial_variable_still_in_base = false;
        for (unsigned basic_variable_row = 0; basic_variable_row != constraints_count; ++basic_variable_row) {
          const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
          if (basic_variable_col >= client_variables_count + slack_variables_count) {
            artificial_variable_still_in_base = true;
            break;
          }
        }
        if (!artificial_variable_still_in_base) {
          return Optimal{};
        }
      }
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
    const auto& objective = tableau[constraints_count + objectives_count - 1];

    std::optional<unsigned> entering_column;
    for (unsigned col = 0; col != total_variables_count; ++col) {
      // @todo Consider adding: if (col < client_variables_count + slack_variables_count)  // Never select artificial variables for entering
      if (objective[col] > 0) {
        if (!entering_column || objective[col] > objective[*entering_column]) {
          entering_column = col;
        }
      }
    }

    return entering_column;
  }

  std::optional<unsigned> find_leaving_row(const unsigned entering_column) const {
    std::optional<unsigned> leaving_row;
    for (unsigned row = 0; row != constraints_count; ++row) {
      // @todo Consider adding: if (artificial_variables_count == 0 || basic_variable_cols[row] >= client_variables_count + slack_variables_count)  // Only select artificial variables for leaving when there are some
      if (tableau[row][entering_column] > 0) {
        if (!leaving_row || tableau[row][total_variables_count] / tableau[row][entering_column] < tableau[*leaving_row][total_variables_count] / tableau[*leaving_row][entering_column]) {
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
    const Tableau::fp_type pivot_value = tableau[leaving_row][entering_column];
    if (verbosity > 2) {
      std::cerr << boost::format("    pivot value: %|-.3|") % pivot_value << std::endl;
    }
    assert(!std::isnan(pivot_value));
    assert(!std::isinf(pivot_value));
    assert(pivot_value > 0);

    for (unsigned row = 0; row != tableau_height; ++row) {
      if (row != leaving_row) {
        const Tableau::fp_type factor = tableau[row][entering_column];
        if (verbosity > 2) {
          std::cerr << boost::format("    %|| %|| factor: %|-.3|") % (row < constraints_count ? "constraint" : "objective") % (row < constraints_count ? variable_name(basic_variable_cols[row]) : std::to_string(row - constraints_count)) % factor << std::endl;
        }
        assert(!std::isnan(factor));
        assert(!std::isinf(factor));
        for (unsigned col = 0; col != tableau_width; ++col) {
          if (col != entering_column) {
            // The goal here is to approximate the following mathematical operation:
            //   tableau[row][col] = tableau[row][col] - factor * tableau[leaving_row][col] / pivot_value
            // We try to avoid numerical instability with the following:
            const Tableau::fp_type numerator = tableau[row][col] * pivot_value - factor * tableau[leaving_row][col];
            // @todo(Feature, later) Investigate how to make this small value relative instead of absolute
            // (I anticipate/fear it won't work well with some kinds of small-valued linear programs)
            // (I also feel the current value is too large to be acceptable in general, but it appears to be necessary in some of our tests)
            if (std::abs(numerator) < 1e-6) {
              tableau[row][col] = 0;
            } else {
              tableau[row][col] = numerator / pivot_value;
            }
          }
        }
        tableau[row][entering_column] = 0;
      }
    }

    for (unsigned col = 0; col != tableau_width; ++col) {
      if (col != entering_column) {
        tableau[leaving_row][col] = tableau[leaving_row][col] / pivot_value;
      }
    }
    tableau[leaving_row][entering_column] = 1;

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
            std::cerr << boost::format("%|=8.3|") % tableau[row][col];
          }
          std::cerr << boost::format("| %|-.3|\n") % tableau[row][total_variables_count];
        }

        std::cerr << "    objectives:\n";
        for (unsigned row = 0; row != objectives_count; ++row) {
          std::cerr << "          |";
          for (unsigned col = 0; col < total_variables_count; ++col) {
            std::cerr << boost::format("%|=8.3|") % tableau[constraints_count + row][col];
          }
          std::cerr << boost::format("| %|-.3|") % tableau[constraints_count + row][total_variables_count] << std::endl;
        }
      } else {
        std::cerr << "    costs:";
        for (unsigned row = 0; row != objectives_count; ++row) {
          std::cerr << " " << tableau[constraints_count + row][total_variables_count];
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
    assert_no_nans_or_infinities();
  }

  void assert_sizes_are_consistent() {
    assert(total_variables_count == client_variables_count + slack_variables_count + artificial_variables_count);
    assert(tableau_width == total_variables_count + 1);
    assert(tableau_height == constraints_count + objectives_count);
    assert(tableau.s1() == tableau_height);
    assert(tableau.s0() == tableau_width);
    assert(basic_variable_cols.size() == constraints_count);
  }

  void assert_basic_variables_are_consistent() {
    for (unsigned basic_variable_row = 0; basic_variable_row != constraints_count; ++basic_variable_row) {
      const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
      for (unsigned row = 1; row < constraints_count; ++row) {
        assert(tableau[row][basic_variable_col] == (row == basic_variable_row ? 1. : 0.));
      }
    }
  }

  void assert_no_nans_or_infinities() {
    for (unsigned row = 0; row != tableau_height; ++row) {
      for (unsigned col = 0; col != tableau_width; ++col) {
        assert(!std::isnan(tableau[row][col]));
        assert(!std::isinf(tableau[row][col]));
      }
    }
  }

 private:
  const unsigned client_variables_count;
  const unsigned slack_variables_count;
  const unsigned artificial_variables_count;
  const unsigned total_variables_count;
  const unsigned tableau_width;
  const unsigned constraints_count;
  const unsigned objectives_count;
  const unsigned tableau_height;
  Array2D<Host, Tableau::fp_type>& tableau;
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
    const unsigned client_variables_count = program.variables_count();
    if (artificial_variables_count == 0) {
      if (verbosity > 1) {
        std::cerr << "SINGLE PHASE" << std::endl;
        std::cerr << "============" << std::endl;
      }
      auto tableau = make_single_phase_tableau(constraints_count, slack_variables_count);
      const auto run_result = Simplex(tableau).run();
      if (verbosity > 1) {
        std::cerr << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(run_result)) {
        if (verbosity > 0) {
          std::cerr << "OPTIMAL (single-phase)" << std::endl;
        }
        return CustomOnCpuLinearProgram::solution_type{tableau.get_assignments(), static_cast<float>(tableau.tableau[tableau.constraints_count + 0][tableau.total_variables_count])};
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
      auto first_tableau = make_first_phase_tableau(constraints_count, slack_variables_count, artificial_variables_count);
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
      {
        // @todo(Feature, later) Understand why we need such a high tolerance; try to reduce it
        const Tableau::fp_type epsilon = 1e-3;
        if (verbosity > 0 || std::abs(first_tableau.tableau[constraints_count + 1][first_tableau.total_variables_count]) >= epsilon) {
          std::cerr << "FEASIBLE. First phase cost (should be zero): " << first_tableau.tableau[constraints_count + 1][first_tableau.total_variables_count] << std::endl;
        }
        assert_with_dump(std::abs(first_tableau.tableau[constraints_count + 1][first_tableau.total_variables_count]) < epsilon, program);
      }

      if (verbosity > 1) {
        std::cerr << "SECOND PHASE" << std::endl;
        std::cerr << "============" << std::endl;
      }

      auto second_tableau = make_second_phase_tableau(first_tableau);
      const auto second_run_result = Simplex(second_tableau).run();
      if (verbosity > 1) {
        std::cerr << std::endl;
      }
      if (std::holds_alternative<Simplex::Optimal>(second_run_result)) {
        if (verbosity > 0) {
          std::cerr << "OPTIMAL (two phases)" << std::endl;
        }
        return CustomOnCpuLinearProgram::solution_type{second_tableau.get_assignments(), static_cast<float>(second_tableau.tableau[constraints_count + 0][client_variables_count + slack_variables_count])};
      } else {
        if (verbosity > 0) {
          std::cerr << "UNBOUNDED (two phases)" << std::endl;
        }
        return {};
      }
    }
  }

 private:
  Tableau make_single_phase_tableau(const unsigned constraints_count, const unsigned slack_variables_count) {
    const unsigned client_variables_count = program.variables_count();
    const unsigned artificial_variables_count = 0;
    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;
    const unsigned tableau_width = total_variables_count + 1;
    const unsigned objectives_count = 1;
    const unsigned tableau_height = constraints_count + objectives_count;

    Array2D<Host, Tableau::fp_type> tableau(tableau_height, tableau_width, zeroed);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau,
      basic_variable_cols,
      artificial_variable_rows
    );

    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      tableau[constraints_count + 0][variable_index] = -coefficient;
    }

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(tableau),
      std::move(basic_variable_cols)
    };
  }

  Tableau make_first_phase_tableau(const unsigned constraints_count, const unsigned slack_variables_count, const unsigned artificial_variables_count) {
    const unsigned client_variables_count = program.variables_count();
    assert(artificial_variables_count != 0);
    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;
    const unsigned tableau_width = total_variables_count + 1;
    const unsigned objectives_count = 2;
    const unsigned tableau_height = constraints_count + objectives_count;

    Array2D<Host, Tableau::fp_type> tableau(tableau_height, tableau_width, zeroed);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau,
      basic_variable_cols,
      artificial_variable_rows
    );

    // Phase 2 objective
    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      tableau[constraints_count + 0][variable_index] = -coefficient;
    }
    // Phase 1 objective...
    for (unsigned artificial_variable_index = client_variables_count + slack_variables_count; artificial_variable_index < total_variables_count; ++artificial_variable_index) {
      tableau[constraints_count + 1][artificial_variable_index] = -1;
    }
    // ... in term of non-basic variables
    for (unsigned artificial_variable_index = 0; artificial_variable_index != artificial_variables_count; ++artificial_variable_index) {
      const unsigned row = artificial_variable_rows[artificial_variable_index];
      for (unsigned col = 0; col < tableau_width; ++col) {
        tableau[constraints_count + 1][col] += tableau[row][col];
      }
    }

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(tableau),
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
    const unsigned total_variables_count,
    Array2D<Host, Tableau::fp_type>& tableau,
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
          tableau[constraint_index][variable_index] = coefficient;
        }
        tableau[constraint_index][total_variables_count] = constraint.upper_bound;

        tableau[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
        basic_variable_cols[constraint_index] = client_variables_count + slack_variables_count + artificial_variable_index;
        artificial_variable_rows[artificial_variable_index] = constraint_index;
        ++artificial_variable_index;

        ++constraint_index;
      } else {
        if (constraint.upper_bound != infinity) {
          if (constraint.upper_bound >= 0) {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              tableau[constraint_index][variable_index] = coefficient;
            }
            tableau[constraint_index][total_variables_count] = constraint.upper_bound;

            tableau[constraint_index][client_variables_count + slack_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;
          } else {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              tableau[constraint_index][variable_index] = -coefficient;
            }
            tableau[constraint_index][total_variables_count] = -constraint.upper_bound;

            tableau[constraint_index][client_variables_count + slack_variable_index] = -1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;

            tableau[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variables_count + artificial_variable_index;
            artificial_variable_rows[artificial_variable_index] = constraint_index;
            ++artificial_variable_index;
          }

          ++constraint_index;
        }
        if (constraint.lower_bound != -infinity) {
          if (constraint.lower_bound <= 0) {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              tableau[constraint_index][variable_index] = -coefficient;
            }
            tableau[constraint_index][total_variables_count] = -constraint.lower_bound;

            tableau[constraint_index][client_variables_count + slack_variable_index] = 1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;
          } else {
            for (const auto& [variable_index, coefficient] : constraint.coefficients) {
              tableau[constraint_index][variable_index] = coefficient;
            }
            tableau[constraint_index][total_variables_count] = constraint.lower_bound;

            tableau[constraint_index][client_variables_count + slack_variable_index] = -1;
            basic_variable_cols[constraint_index] = client_variables_count + slack_variable_index;
            ++slack_variable_index;

            tableau[constraint_index][client_variables_count + slack_variables_count + artificial_variable_index] = 1;
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

  Tableau make_second_phase_tableau(const Tableau& first_tableau) {
    // @todo Consider reusing the first tableau's memory instead of copying it.
    // It's not clear a priori what's the most costly:
    //   - the copy to create a contiguous second phase tableau
    //   - the reduction of cache locality of using a non-contiguous second phase tableau

    const unsigned client_variables_count = first_tableau.client_variables_count;
    const unsigned slack_variables_count = first_tableau.slack_variables_count;
    const unsigned artificial_variables_count = 0;
    const unsigned total_variables_count = client_variables_count + slack_variables_count + artificial_variables_count;
    const unsigned tableau_width = total_variables_count + 1;
    const unsigned constraints_count = first_tableau.constraints_count;
    const unsigned objectives_count = 1;
    const unsigned tableau_height = constraints_count + objectives_count;

    Array2D<Host, Tableau::fp_type> tableau(tableau_height, tableau_width, zeroed);
    for (unsigned variable_index = 0; variable_index != total_variables_count; ++variable_index) {
      tableau[constraints_count + 0][variable_index] = first_tableau.tableau[constraints_count + 0][variable_index];
    }
    tableau[constraints_count + 0][total_variables_count] = first_tableau.tableau[constraints_count + 0][first_tableau.total_variables_count];

    for (unsigned row = 0; row != constraints_count; ++row) {
      for (unsigned col = 0; col != total_variables_count; ++col) {
        tableau[row][col] = first_tableau.tableau[row][col];
      }
    }
    for (unsigned row = 0; row != constraints_count; ++row) {
      tableau[row][total_variables_count] = first_tableau.tableau[row][first_tableau.total_variables_count];
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
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(tableau),
      std::move(basic_variable_cols)
    };
  };

 private:
  const CustomOnCpuLinearProgram& program;
};

std::optional<CustomOnCpuLinearProgram::solution_type> CustomOnCpuLinearProgram::solve() {
  const auto solution = CustomOnCpuLinearProgramSolver(*this).solve();
  #ifndef NDEBUG
  const auto glop_solution = glop_program.solve();
  assert_with_dump(glop_solution.has_value() == solution.has_value(), *this);
  if (solution) {
    assert_with_dump(!std::isnan(glop_solution->cost), *this);
    assert_with_dump(!std::isinf(glop_solution->cost), *this);
    assert_with_dump(!std::isnan(solution->cost), *this);
    assert_with_dump(!std::isinf(solution->cost), *this);
    {
      // @todo(Feature, later) Figure out a good predictable way to determine this epsilon
      // (I feel like this one is too large to be acceptable in general, but it appears to be necessary in some of our tests)
      const Tableau::fp_type epsilon = 1e-2;
      const bool close_enough =
        (std::abs(glop_solution->cost) < epsilon && std::abs(solution->cost) < epsilon)
        ||
        std::abs(glop_solution->cost - solution->cost) / std::max(std::abs(glop_solution->cost), std::abs(solution->cost)) < epsilon;
      if (!close_enough) {
        std::cerr << "Glop: " << glop_solution->cost << " vs. Custom: " << solution->cost
          << "(rel. diff.:" << std::abs(glop_solution->cost - solution->cost) / std::max(std::abs(glop_solution->cost), std::abs(solution->cost)) << ")" << std::endl;
      }
      assert_with_dump(close_enough, *this);
    }
  }
  #endif
  return solution;
}

}  // namespace lincs
