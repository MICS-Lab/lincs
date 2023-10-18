// Copyright 2023 Vincent Jacques

#ifndef LINCS__RANDOMNESS_UTILS_HPP
#define LINCS__RANDOMNESS_UTILS_HPP

#include <algorithm>
#include <cassert>
#include <map>
#include <random>
#include <vector>


/*
Pick random values from a finite set with given probabilities
(a discrete distribution with arbitrary values).
*/
template<typename T>
class ProbabilityWeightedGenerator {
  ProbabilityWeightedGenerator(const std::vector<T>& values_, const std::vector<double>& probabilities) :
    values(values_),
    distribution(probabilities.begin(), probabilities.end())
  {}

  // I tried using ranges and all but I failed
  std::vector<T> map_keys(const std::map<T, double>& value_probabilities) {
    std::vector<T> keys;
    for (const auto& [k, v] : value_probabilities) {
      keys.push_back(k);
    }
    return keys;
  }

  std::vector<double> map_values(const std::map<T, double>& value_probabilities) {
    std::vector<double> values;
    for (const auto& [k, v] : value_probabilities) {
      values.push_back(v);
    }
    return values;
  }

 public:
  ProbabilityWeightedGenerator(const std::map<T, double>& value_probabilities) :
    ProbabilityWeightedGenerator(map_keys(value_probabilities), map_values(value_probabilities))
  {
    assert(!value_probabilities.empty());
  }

  template<typename Generator>
  T operator()(Generator& gen) const {
    const unsigned index = distribution(gen);
    assert(index < values.size());
    return values[index];
  }

 private:
  std::vector<T> values;
  mutable std::discrete_distribution<unsigned> distribution;
};

template<typename T>
void shuffle(std::mt19937& urbg, T m) {
  for (unsigned i = 0; i != m.s0(); ++i) {
    std::swap(m[i], m[std::uniform_int_distribution<unsigned>(0, m.s0() - 1)(urbg)]);
  }
}

#endif  // LINCS__RANDOMNESS_UTILS_HPP
