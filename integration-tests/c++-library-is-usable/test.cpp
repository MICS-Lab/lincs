// Copyright 2023 Vincent Jacques

#include <iostream>
#include <sstream>

#include <lincs.hpp>


// @todo Remove details from this example, turn them into unit tests (see ../python-package-is-usable/test.py)

int main() {
  lincs::Domain domain{
    {
      {"Literature grade", lincs::Domain::Criterion::ValueType::real, lincs::Domain::Criterion::CategoryCorrelation::growing},
      {"Physics grade", lincs::Domain::Criterion::ValueType::real, lincs::Domain::Criterion::CategoryCorrelation::growing},
    },
    {
      {"Fail"},
      {"Pass"},
    }
  };

  domain.dump(std::cout);

  lincs::Model model{domain, {{{10.f, 10.f}, {lincs::Model::SufficientCoalitions::Kind::weights, {0.4f, 0.7f}}}}};
  {
    std::ostringstream oss;
    model.dump(oss);
    std::cout << oss.str() << std::endl;
    std::istringstream iss(oss.str());
    lincs::Model model2 = lincs::Model::load(domain, iss);

    model2.dump(std::cout);
  }

  lincs::Alternatives alternatives{domain, {{"Alice", {11.f, 12.f}, "Pass"}, {"Bob", {9.f, 11.f}, "Fail"}}};
  {
    std::ostringstream oss;
    alternatives.dump(oss);
    std::cout << oss.str();
    std::istringstream iss(oss.str());
    lincs::Alternatives alternatives2 = lincs::Alternatives::load(domain, iss);

    alternatives2.dump(std::cout);
  }
}
