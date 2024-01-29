// Copyright 2023-2024 Vincent Jacques

#include <iostream>

#include <lincs.hpp>


int main() {
  lincs::Problem problem{
    {
      lincs::Criterion("Physics grade", lincs::Criterion::RealValues(lincs::Criterion::PreferenceDirection::increasing, 0, 1)),
      lincs::Criterion("Literature grade", lincs::Criterion::RealValues(lincs::Criterion::PreferenceDirection::increasing, 0, 1)),
    },
    {
      {"Bad"},
      {"Good"},
    }
  };

  problem.dump(std::cout);

  std::cout << "\n";

  lincs::Model model{
    problem,
    {
      lincs::AcceptedValues(lincs::AcceptedValues::RealThresholds({10.f})),
      lincs::AcceptedValues(lincs::AcceptedValues::RealThresholds({10.f})),
    },
    {lincs::SufficientCoalitions(lincs::SufficientCoalitions::Weights({0.4f, 0.7f}))},
  };
  model.dump(problem, std::cout);

  std::cout << "\n";

  lincs::Alternatives alternatives{
    problem,
    {
      {
        "Alice",
        {
          lincs::Performance(lincs::Performance::Real(11)),
          lincs::Performance(lincs::Performance::Real(12)),
        },
        1
      },
      {
        "Bob",
        {
          lincs::Performance(lincs::Performance::Real(9)),
          lincs::Performance(lincs::Performance::Real(11)),
        },
        0
      },
    },
  };
  alternatives.dump(problem, std::cout);
}
