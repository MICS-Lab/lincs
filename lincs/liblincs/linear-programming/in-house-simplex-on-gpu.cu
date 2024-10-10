// Copyright 2024 Vincent Jacques

#include "in-house-simplex-on-gpu.hpp"

#include <cmath>
#include <variant>

#include "../vendored/lov-e.hpp"

namespace lincs {

namespace {

typedef GridFactory2D<16, 16> tableau_grid;
typedef GridFactory1D<256> column_grid;
typedef GridFactory1D<256> row_grid;

constexpr unsigned index_not_found = std::numeric_limits<unsigned>::max();
constexpr float infinity = std::numeric_limits<float>::infinity();

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
  Array2D<Host, fp_type> host_tableau;
  Array2D<Device, fp_type> device_tableau;
  std::vector<unsigned> basic_variable_cols;

  std::vector<float> get_assignments_from_host() const {
    std::vector<float> assignments(total_variables_count, 0);
    for (unsigned basic_variable_row = 0; basic_variable_row < constraints_count; ++basic_variable_row) {
      const unsigned basic_variable_col = basic_variable_cols[basic_variable_row];
      assignments[basic_variable_col] = host_tableau[basic_variable_row][total_variables_count];
    }
    return assignments;
  }
};

__global__
void find_entering_column_index__kernel(
  const ArrayView2D<Device, const Tableau::fp_type> tableau,
  const ArrayView1D<Device, unsigned> indexes
) {
  const unsigned objective_row_index = tableau.s1() - 1;

  unsigned& entering_column_index = indexes[1];
  entering_column_index = index_not_found;
  // @todo Do this search in parallel, obviously
  for (unsigned column_index = 0; column_index != tableau.s0() - 1; ++column_index) {
    if (tableau[objective_row_index][column_index] > 0) {
      if (
        entering_column_index == index_not_found
        || tableau[objective_row_index][column_index] > tableau[objective_row_index][entering_column_index]
      ) {
        entering_column_index = column_index;
      }
    }
  }
}

__global__
void find_leaving_row_index__kernel(
  const ArrayView2D<Device, const Tableau::fp_type> tableau,
  const unsigned constraints_count,
  const ArrayView1D<Device, unsigned> indexes
) {
  const unsigned variable_values_column_index = tableau.s0() - 1;
  const unsigned entering_column_index = indexes[1];
  unsigned& leaving_row_index = indexes[0];
  leaving_row_index = index_not_found;
  // @todo Do this search in parallel, obviously
  for (unsigned row_index = 0; row_index != constraints_count; ++row_index) {
    if (tableau[row_index][entering_column_index] > 0) {
      if (
        leaving_row_index == index_not_found
        || tableau[row_index][variable_values_column_index] / tableau[row_index][entering_column_index] < tableau[leaving_row_index][variable_values_column_index] / tableau[leaving_row_index][entering_column_index]
      ) {
        leaving_row_index = row_index;
      }
    }
  }
}

__global__
void extract_entering_column__kernel(
  const ArrayView2D<Device, const Tableau::fp_type> tableau,
  const ArrayView1D<Device, const unsigned> indexes,
  const ArrayView1D<Device, Tableau::fp_type> entering_column
) {
  assert(tableau.s1() == entering_column.s0());
  assert(indexes.s0() == 2);

  const unsigned row = column_grid::x();
  assert(row < tableau.s1() + column_grid::blockDim().x);
  const unsigned entering_column_index = indexes[1];
  assert(entering_column_index < tableau.s0());

  if (row < tableau.s1()) {
    entering_column[row] = tableau[row][entering_column_index];
  }
}

__global__
void extract_leaving_row__kernel(
  const ArrayView2D<Device, const Tableau::fp_type> tableau,
  const ArrayView1D<Device, const unsigned> indexes,
  const ArrayView1D<Device, Tableau::fp_type> leaving_row
) {
  assert(tableau.s0() == leaving_row.s0());
  assert(indexes.s0() == 2);

  const unsigned leaving_row_index = indexes[0];
  assert(leaving_row_index < tableau.s1());
  const unsigned col = row_grid::x();
  assert(col < tableau.s0() + row_grid::blockDim().x);

  if (col < tableau.s0()) {
    leaving_row[col] = tableau[leaving_row_index][col];
  }
}

__global__
void pivot__kernel(
  const ArrayView1D<Device, const unsigned> indexes,
  const ArrayView1D<Device, const Tableau::fp_type> leaving_row,
  const ArrayView1D<Device, const Tableau::fp_type> entering_column,
  const ArrayView2D<Device, Tableau::fp_type> tableau
) {
  assert(indexes.s0() == 2);
  const unsigned leaving_row_index = indexes[0];
  const unsigned entering_column_index = indexes[1];

  assert(tableau.s0() == leaving_row.s0());
  assert(leaving_row_index < tableau.s1());
  const unsigned row = tableau_grid::x();
  assert(row < tableau.s1() + tableau_grid::blockDim().x);

  assert(tableau.s1() == entering_column.s0());
  assert(entering_column_index < tableau.s0());
  const unsigned col = tableau_grid::y();
  assert(col < tableau.s0() + tableau_grid::blockDim().y);

  const Tableau::fp_type pivot_value = leaving_row[entering_column_index];

  if (row < tableau.s1() && col < tableau.s0()) {
    const Tableau::fp_type numerator = tableau[row][col] * pivot_value - entering_column[row] * leaving_row[col];
    const bool zero = col == entering_column_index || std::abs(numerator) < 1e-6;
    tableau[row][col] = row == leaving_row_index ? leaving_row[col] / pivot_value : (zero ? 0 : numerator / pivot_value);
  }
}

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
    host_tableau(tableau.host_tableau),
    device_tableau(tableau.device_tableau),
    tableau_grid(tableau_grid::make(tableau_height, tableau_width)),
    host_indexes(2, uninitialized),
    entering_column_index(host_indexes[1]),
    leaving_row_index(host_indexes[0]),
    device_indexes(2, uninitialized),
    device_entering_column(tableau_height, uninitialized),
    column_grid(column_grid::make(tableau_height)),
    device_leaving_row(tableau_width, uninitialized),
    row_grid(row_grid::make(tableau_width)),
    basic_variable_cols(tableau.basic_variable_cols)
  {}

 public:
  enum class RunResult { optimal, unbounded };

 private:
  enum class StepResult { optimal, unbounded, pivot };

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
          return RunResult::optimal;
        }
      }
      if (step_result == StepResult::optimal) {
        return RunResult::optimal;
      } else if (step_result == StepResult::unbounded) {
        return RunResult::unbounded;
      } else {
        assert(step_result == StepResult::pivot);
      }
    }
  }

 private:
  StepResult step() {
    find_entering_column_index();
    if (entering_column_index != index_not_found) {
      find_leaving_row_index();
      if (leaving_row_index != index_not_found) {
        pivot();
        return StepResult::pivot;
      } else {
        return StepResult::unbounded;
      }
    } else {
      return StepResult::optimal;
    }
  }

  void find_entering_column_index() const {
    find_entering_column_index__kernel<<<1, 1>>>(device_tableau, ref(device_indexes));
    copy(device_indexes, ref(host_indexes));
  }

  void find_leaving_row_index() const {
    find_leaving_row_index__kernel<<<1, 1>>>(device_tableau, constraints_count, ref(device_indexes));
    copy(device_indexes, ref(host_indexes));
  }

  void pivot() {
    extract_entering_column__kernel<<<LOVE_CONFIG(column_grid)>>>(device_tableau, device_indexes, ref(device_entering_column));
    extract_leaving_row__kernel<<<LOVE_CONFIG(row_grid)>>>(device_tableau, device_indexes, ref(device_leaving_row));
    pivot__kernel<<<LOVE_CONFIG(tableau_grid)>>>(device_indexes, device_leaving_row, device_entering_column, ref(device_tableau));
    basic_variable_cols[leaving_row_index] = entering_column_index;
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
  Array2D<Host, Tableau::fp_type>& host_tableau;
  Array2D<Device, Tableau::fp_type>& device_tableau;
  const Array1D<Host, unsigned> host_indexes;
  const unsigned& entering_column_index;
  const unsigned& leaving_row_index;
  const Array1D<Device, unsigned> device_indexes;
  Grid tableau_grid;
  Array1D<Device, Tableau::fp_type> device_entering_column;
  Grid column_grid;
  Array1D<Device, Tableau::fp_type> device_leaving_row;
  Grid row_grid;
  std::vector<unsigned>& basic_variable_cols;
};

class InHouseSimplexOnGpuLinearProgramSolver {
 public:
  InHouseSimplexOnGpuLinearProgramSolver(const InHouseSimplexOnGpuLinearProgram& program) : program(program) {}

 public:
  std::optional<InHouseSimplexOnGpuLinearProgram::solution_type> solve() {
    const auto [constraints_count, slack_variables_count, artificial_variables_count] = count_constraints_and_additional_variables();
    const unsigned client_variables_count = program.variables_count();
    if (artificial_variables_count == 0) {
      auto tableau = make_single_phase_tableau(constraints_count, slack_variables_count);
      const auto run_result = Simplex(tableau).run();
      if (run_result == Simplex::RunResult::optimal) {
        copy(tableau.device_tableau, ref(tableau.host_tableau));
        return InHouseSimplexOnGpuLinearProgram::solution_type{tableau.get_assignments_from_host(), static_cast<float>(tableau.host_tableau[tableau.constraints_count + 0][tableau.total_variables_count])};
      } else {
        return {};
      }
    } else {
      auto first_tableau = make_first_phase_tableau(constraints_count, slack_variables_count, artificial_variables_count);
      Simplex(first_tableau).run();  // We don't care if it's optimal or unbounded

      copy(first_tableau.device_tableau, ref(first_tableau.host_tableau));
      const auto first_assignments = first_tableau.get_assignments_from_host();
      for (unsigned artificial_variable_index = 0; artificial_variable_index < first_tableau.artificial_variables_count; ++artificial_variable_index) {
        if (first_assignments[first_tableau.client_variables_count + first_tableau.slack_variables_count + artificial_variable_index] != 0) {
          return {};
        }
      }

      auto second_tableau = make_second_phase_tableau(first_tableau);
      const auto second_run_result = Simplex(second_tableau).run();
      if (second_run_result == Simplex::RunResult::optimal) {
        copy(second_tableau.device_tableau, ref(second_tableau.host_tableau));
        return InHouseSimplexOnGpuLinearProgram::solution_type{second_tableau.get_assignments_from_host(), static_cast<float>(second_tableau.host_tableau[constraints_count + 0][client_variables_count + slack_variables_count])};
      } else {
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

    Array2D<Host, Tableau::fp_type> host_tableau(tableau_height, tableau_width, zeroed);
    Array2D<Device, Tableau::fp_type> device_tableau(tableau_height, tableau_width, uninitialized);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      host_tableau,
      basic_variable_cols,
      artificial_variable_rows
    );

    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      host_tableau[constraints_count + 0][variable_index] = -coefficient;
    }

    copy(host_tableau, ref(device_tableau));

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(host_tableau),
      std::move(device_tableau),
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

    Array2D<Host, Tableau::fp_type> host_tableau(tableau_height, tableau_width, zeroed);
    Array2D<Device, Tableau::fp_type> device_tableau(tableau_height, tableau_width, uninitialized);
    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    std::vector<unsigned> artificial_variable_rows(total_variables_count, 0);
    populate_constraints(
      constraints_count,
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      host_tableau,
      basic_variable_cols,
      artificial_variable_rows
    );

    // Phase 2 objective
    for (const auto& [variable_index, coefficient] : program.get_objective_coefficients()) {
      host_tableau[constraints_count + 0][variable_index] = -coefficient;
    }
    // Phase 1 objective...
    for (unsigned artificial_variable_index = client_variables_count + slack_variables_count; artificial_variable_index < total_variables_count; ++artificial_variable_index) {
      host_tableau[constraints_count + 1][artificial_variable_index] = -1;
    }
    // ... in term of non-basic variables
    for (unsigned artificial_variable_index = 0; artificial_variable_index != artificial_variables_count; ++artificial_variable_index) {
      const unsigned row = artificial_variable_rows[artificial_variable_index];
      for (unsigned col = 0; col < tableau_width; ++col) {
        host_tableau[constraints_count + 1][col] += host_tableau[row][col];
      }
    }

    copy(host_tableau, ref(device_tableau));

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(host_tableau),
      std::move(device_tableau),
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

    Array2D<Host, Tableau::fp_type> host_tableau(tableau_height, tableau_width, zeroed);
    Array2D<Device, Tableau::fp_type> device_tableau(tableau_height, tableau_width, uninitialized);
    for (unsigned variable_index = 0; variable_index != total_variables_count; ++variable_index) {
      host_tableau[constraints_count + 0][variable_index] = first_tableau.host_tableau[constraints_count + 0][variable_index];
    }
    host_tableau[constraints_count + 0][total_variables_count] = first_tableau.host_tableau[constraints_count + 0][first_tableau.total_variables_count];

    for (unsigned row = 0; row != constraints_count; ++row) {
      for (unsigned col = 0; col != total_variables_count; ++col) {
        host_tableau[row][col] = first_tableau.host_tableau[row][col];
      }
    }
    for (unsigned row = 0; row != constraints_count; ++row) {
      host_tableau[row][total_variables_count] = first_tableau.host_tableau[row][first_tableau.total_variables_count];
    }

    std::vector<unsigned> basic_variable_cols(constraints_count, 0);
    for (unsigned row = 0; row != constraints_count; ++row) {
      basic_variable_cols[row] = first_tableau.basic_variable_cols[row];
    }

    copy(host_tableau, ref(device_tableau));

    return {
      client_variables_count,
      slack_variables_count,
      artificial_variables_count,
      total_variables_count,
      tableau_width,
      constraints_count,
      objectives_count,
      tableau_height,
      std::move(host_tableau),
      std::move(device_tableau),
      std::move(basic_variable_cols)
    };
  };

 private:
  const InHouseSimplexOnGpuLinearProgram& program;
};

}  // namespace

std::optional<InHouseSimplexOnGpuLinearProgram::solution_type> InHouseSimplexOnGpuLinearProgram::solve() {
  const auto solution = InHouseSimplexOnGpuLinearProgramSolver(*this).solve();

  #ifndef NDEBUG
  const auto on_cpu_solution = on_cpu_program.solve();
  assert(solution.has_value() == on_cpu_solution.has_value());
  if (on_cpu_solution.has_value()) {
    assert(!std::isnan(on_cpu_solution->cost));
    assert(!std::isinf(on_cpu_solution->cost));
    assert(!std::isnan(solution->cost));
    assert(!std::isinf(solution->cost));
    {
      const Tableau::fp_type epsilon = 1e-5;
      const bool close_enough =
        (std::abs(on_cpu_solution->cost) < epsilon && std::abs(solution->cost) < epsilon)
        ||
        std::abs(on_cpu_solution->cost - solution->cost) / std::max(std::abs(on_cpu_solution->cost), std::abs(solution->cost)) < epsilon;
      if (!close_enough) {
        std::cerr << "Cost found on CPU: " << on_cpu_solution->cost << " vs. GPU: " << solution->cost
          << " (rel. diff.:" << std::abs(on_cpu_solution->cost - solution->cost) / std::max(std::abs(on_cpu_solution->cost), std::abs(solution->cost)) << ")" << std::endl;
      }
      assert(close_enough);
    }
  }
  #endif

  return solution;
}

}  // namespace lincs
