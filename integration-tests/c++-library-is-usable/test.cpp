// Copyright 2023 Vincent Jacques

#include <iostream>
#include <sstream>

#include <lincs.hpp>


// @todo Remove details from this example, turn them into unit tests (see ../python-package-is-usable/test.py)

int main() {
  lincs::Problem problem{
    {
      {"Literature grade", lincs::Criterion::ValueType::real, lincs::Criterion::CategoryCorrelation::growing},
      {"Physics grade", lincs::Criterion::ValueType::real, lincs::Criterion::CategoryCorrelation::growing},
    },
    {
      {"Fail"},
      {"Pass"},
    }
  };

  problem.dump(std::cout);

  lincs::Model model{problem, {{{10.f, 10.f}, {lincs::SufficientCoalitions::weights, {0.4f, 0.7f}}}}};
  {
    std::ostringstream oss;
    model.dump(problem, oss);
    std::cout << oss.str() << std::endl;
    std::istringstream iss(oss.str());
    lincs::Model model2 = lincs::Model::load(problem, iss);

    model2.dump(problem, std::cout);
  }

  lincs::Alternatives alternatives{problem, {{"Alice", {11.f, 12.f}, 1}, {"Bob", {9.f, 11.f}, 0}}};
  {
    std::ostringstream oss;
    alternatives.dump(problem, oss);
    std::cout << oss.str();
    std::istringstream iss(oss.str());
    lincs::Alternatives alternatives2 = lincs::Alternatives::load(problem, iss);

    alternatives2.dump(problem, std::cout);
  }
}
