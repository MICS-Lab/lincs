// Copyright 2023 Vincent Jacques

#include "randomness-utils.hpp"

#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


TEST_CASE("ProbabilityWeightedGenerator") {
  std::map<std::string, double> value_probabilities = {
    {"a", 0.1},
    {"b", 0.2},
    {"c", 0.3},
    {"d", 0.4},
  };

  auto generator = ProbabilityWeightedGenerator(value_probabilities);

  std::mt19937 mt(42);
  std::map<std::string, unsigned> counts;
  for (unsigned i = 0; i < 10000; ++i) {
    const std::string value = generator(mt);
    counts[value] += 1;
  }

  CHECK(counts.size() == 4);
  CHECK(counts["a"] == 1016);
  CHECK(counts["b"] == 1958);
  CHECK(counts["c"] == 3002);
  CHECK(counts["d"] == 4024);
}

TEST_CASE("Equivalent 'ProbabilityWeightedGenerator's with copied generators generate equivalent values") {
  std::mt19937 mt1(42);
  std::mt19937 mt2(mt1);

  auto generator1 = ProbabilityWeightedGenerator<int>({
    {0, 0.1},
    {1, 0.2},
    {2, 0.3},
    {3, 0.4},
  });

  auto generator2 = ProbabilityWeightedGenerator<char>({
    {'3', 0.4},
    {'2', 0.3},
    {'1', 0.2},
    {'0', 0.1},
  });

  CHECK(generator1(mt1) == 3);
  CHECK(generator1(mt1) == 1);
  CHECK(generator1(mt1) == 3);
  CHECK(generator1(mt1) == 2);
  CHECK(generator1(mt1) == 2);
  CHECK(generator1(mt1) == 0);
  CHECK(generator1(mt1) == 2);
  CHECK(generator1(mt1) == 2);
  CHECK(generator1(mt1) == 1);
  CHECK(generator1(mt1) == 3);

  CHECK(generator2(mt2) == '3');
  CHECK(generator2(mt2) == '1');
  CHECK(generator2(mt2) == '3');
  CHECK(generator2(mt2) == '2');
  CHECK(generator2(mt2) == '2');
  CHECK(generator2(mt2) == '0');
  CHECK(generator2(mt2) == '2');
  CHECK(generator2(mt2) == '2');
  CHECK(generator2(mt2) == '1');
  CHECK(generator2(mt2) == '3');
}
