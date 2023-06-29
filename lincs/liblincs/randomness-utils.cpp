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

  std::mt19937 gen(42);
  std::map<std::string, unsigned> counts;
  for (unsigned i = 0; i < 10000; ++i) {
    const std::string value = generator(gen);
    counts[value] += 1;
  }

  CHECK(counts.size() == 4);
  CHECK(counts["a"] == 1016);
  CHECK(counts["b"] == 1958);
  CHECK(counts["c"] == 3002);
  CHECK(counts["d"] == 4024);
}
