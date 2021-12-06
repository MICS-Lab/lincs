// Copyright 2021 Vincent Jacques

#ifndef RANDOMNESS_HPP_
#define RANDOMNESS_HPP_


#include <curand_kernel.h>

#include <map>
#include <random>
#include <vector>

#include "cuda-utils.hpp"


struct RandomSource {
  RandomSource();
  ~RandomSource();

  void init_for_host(int seed);

  void init_for_device(int seed);

  curandState* rng_states;
  std::mt19937* gen;
};

struct RandomNumberGenerator {
  RandomNumberGenerator(const RandomSource& source) :  // NOLINT(runtime/explicit)
    _rng_states(source.rng_states), _gen(source.gen) {}

  __host__ __device__
  float uniform_float(const float min, const float max);

  __host__ __device__
  uint uniform_int(const uint min, const uint max);

  auto& urbg() { return *_gen; }

 private:
  curandState* _rng_states;
  std::mt19937* _gen;
};

template<typename T>
class ProbabilityWeightedGenerator {
  ProbabilityWeightedGenerator(const std::vector<T>& values, const std::vector<double>& probabilities) :
    _values(values),
    _distribution(probabilities.begin(), probabilities.end())
  {}

 public:
  static ProbabilityWeightedGenerator make(std::map<T, double> value_probabilities) {
    std::vector<T> values;
    values.reserve(value_probabilities.size());
    std::vector<double> probabilities;
    probabilities.reserve(value_probabilities.size());
    for (auto value_probability : value_probabilities) {
      values.push_back(value_probability.first);
      probabilities.push_back(value_probability.second);
    }
    return ProbabilityWeightedGenerator(values, probabilities);
  }

  std::map<T, double> get_value_probabilities() {
    std::map<T, double> value_probabilities;
    auto probabilities = _distribution.probabilities();
    const uint size = _values.size();
    assert(probabilities.size() == size);
    for (uint i = 0; i != size; ++i) {
      value_probabilities[_values[i]] = probabilities[i];
    }
    return value_probabilities;
  }

  template<typename Generator>
  T operator()(Generator& gen) const {  // NOLINT(runtime/references)
    const uint index = _distribution(gen);
    assert(index < _values.size());
    return _values[index];
  }

 private:
  std::vector<T> _values;
  mutable std::discrete_distribution<uint> _distribution;
};


#endif  // RANDOMNESS_HPP_
