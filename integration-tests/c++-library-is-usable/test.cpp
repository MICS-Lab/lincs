// Copyright 2023 Vincent Jacques

#include <iostream>

#include <lincs.hpp>


int main() {
  lincs::Problem problem{
    {
      lincs::Criterion::make_real("Physics grade", lincs::Criterion::PreferenceDirection::increasing, 0, 1),
      lincs::Criterion::make_real("Literature grade", lincs::Criterion::PreferenceDirection::increasing, 0, 1),
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
      lincs::AcceptedValues::make_real_thresholds({10.f}),
      lincs::AcceptedValues::make_real_thresholds({10.f}),
    },
    {lincs::SufficientCoalitions::make_weights({0.4f, 0.7f})},
  };
  model.dump(problem, std::cout);

  std::cout << "\n";

  lincs::Alternatives alternatives{problem, {{"Alice", {11.f, 12.f}, 1}, {"Bob", {9.f, 11.f}, 0}}};
  alternatives.dump(problem, std::cout);
}
