// Copyright 2024 Vincent Jacques

#include <cassert>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>

#include "custom-on-cpu.hpp"
#include "../vendored/lov-e.hpp"


namespace lincs {

void print(std::ostream& out, const Array2D<Host, float>& array) {
  for (unsigned i = 0; i < array.s1(); ++i) {
    for (unsigned j = 0; j < array.s0(); ++j) {
      out << array[i][j] << " ";
    }
    out << std::endl;
  }
}

void print(std::ostream& out, const std::map<CustomOnCpuLinearProgram::variable_type, float>& map) {
  for (const auto& [variable, coefficient] : map) {
    out << std::showpos << coefficient << "*x" << variable << " ";
  }
}

std::optional<unsigned> find_pivot_column(const Array2D<Host, float>& tableau) {
  unsigned pivot_column = 1;
  for (unsigned j = 2; j != tableau.s0() - 1; ++j) {
    if (tableau[0][j] > tableau[0][pivot_column]) {
      pivot_column = j;
    }
  }

  if (tableau[0][pivot_column] > 0) {
    return pivot_column;
  } else {
    return {};
  }
}

std::optional<unsigned> find_pivot_row(const Array2D<Host, float>& tableau, unsigned pivot_column) {
  unsigned pivot_row = 1;
  for (unsigned i = 2; i != tableau.s1(); ++i) {
    if (tableau[i][pivot_column] > 0) {
      if (tableau[i][tableau.s0() - 1] / tableau[i][pivot_column] < tableau[pivot_row][tableau.s0() - 1] / tableau[pivot_row][pivot_column]) {
        pivot_row = i;
      }
    }
  }

  if (tableau[pivot_row][pivot_column] > 0) {
    return pivot_row;
  } else {
    return {};
  }
}

void pivot(Array2D<Host, float>& tableau, unsigned pivot_row, unsigned pivot_column) {
  const unsigned s0 = tableau.s0();
  const unsigned s1 = tableau.s1();

  const float pivot_value = tableau[pivot_row][pivot_column];
  for (unsigned j = 0; j != s0; ++j) {
    tableau[pivot_row][j] /= pivot_value;
  }

  for (unsigned i = 0; i != s1; ++i) {
    if (i != pivot_row) {
      const float factor = tableau[i][pivot_column];
      for (unsigned j = 0; j != s0; ++j) {
        tableau[i][j] -= factor * tableau[pivot_row][j];
      }
    }
  }
}

CustomOnCpuLinearProgram::solution_type CustomOnCpuLinearProgram::solve() {
  const float infinity = std::numeric_limits<float>::infinity();

  const unsigned client_variables_count = next_variable_index;

  std::cout << "Objective: minimize  ";
  print(std::cout, objective_coefficients);
  std::cout << std::endl;
  
  std::cout << "Subject to:" << std::endl;
  unsigned slack_variables_count = 0;
  for (const auto& [lower_bound, upper_bound, coefficients] : constraints) {
    assert(upper_bound != infinity || lower_bound != -infinity);
    if (upper_bound != infinity) {
      print(std::cout, coefficients);
      std::cout << " <= " << upper_bound << std::endl;
      ++slack_variables_count;
    }
    if (lower_bound != -infinity) {
      print(std::cout, coefficients);
      std::cout << " >= " << lower_bound << std::endl;
      ++slack_variables_count;
    }
  }

  Array2D<Host, float> tableau(1 + slack_variables_count, 1 + client_variables_count + slack_variables_count + 1, zeroed);

  tableau[0][0] = 1;
  for (unsigned i = 0; i < client_variables_count; ++i) {
    tableau[0][i + 1] = -objective_coefficients[i];
  }

  unsigned next_slack_variable_index = 0;
  for (const auto& [lower_bound, upper_bound, coefficients] : constraints) {
    assert(upper_bound != infinity || lower_bound != -infinity);
    if (upper_bound != infinity) {
      const unsigned slack_variable_index = next_slack_variable_index;
      ++next_slack_variable_index;

      const unsigned row = 1 + slack_variable_index;
      for (const auto& [client_variable_index, coefficient] : coefficients) {
        tableau[row][1 + client_variable_index] = coefficient;
      }
      tableau[row][1 + client_variables_count + slack_variable_index] = 1;
      tableau[row][1 + client_variables_count + slack_variables_count] = upper_bound;
    }
    if (lower_bound != -infinity) {
      const unsigned slack_variable_index = next_slack_variable_index;
      ++next_slack_variable_index;

      const unsigned row = 1 + slack_variable_index;
      for (const auto& [client_variable_index, coefficient] : coefficients) {
        tableau[row][1 + client_variable_index] = -coefficient;
      }
      tableau[row][1 + client_variables_count + slack_variable_index] = 1;
      tableau[row][1 + client_variables_count + slack_variables_count] = -lower_bound;
    }
  }

  std::cout << "Initial tableau:" << std::endl;
  print(std::cout, tableau);

  while (true) {
    std::optional<unsigned> pivot_column = find_pivot_column(tableau);
    if (pivot_column) {
      std::optional<unsigned> pivot_row = find_pivot_row(tableau, *pivot_column);
      if (pivot_row) {
        std::cout << "Tableau after pivot (column: " << *pivot_column << ", row: " << *pivot_row << ')' << std::endl;
        pivot(tableau, *pivot_row, *pivot_column);
        print(std::cout, tableau);
      } else {
        // Unbounded
        assert(false);  // @todo(Feature, soon) Implement unbounded case
      }
    } else {
      // Optimal
      break;
    }
  }

  std::vector<float> assignments(client_variables_count, 0);
  for (unsigned i = 0; i < client_variables_count; ++i) {
    for (unsigned j = 1; j < 1 + slack_variables_count; ++j) {
      if (tableau[j][1 + i] == 1) {
        assignments[i] = tableau[j][1 + client_variables_count + slack_variables_count];
      }
    }
  }
  return {assignments, tableau[0][1 + client_variables_count + slack_variables_count]};
}

}  // namespace lincs
