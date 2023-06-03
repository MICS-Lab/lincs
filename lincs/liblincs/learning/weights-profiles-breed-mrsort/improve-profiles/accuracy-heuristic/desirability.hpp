// Copyright 2023 Vincent Jacques

#ifndef LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC__DESIRABILITY_HPP
#define LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC__DESIRABILITY_HPP

#include <lov-e.hpp>


namespace lincs {

struct Desirability {
  static constexpr float zero_value = 0;

  unsigned v = 0;
  unsigned w = 0;
  unsigned q = 0;
  unsigned r = 0;
  unsigned t = 0;

  #ifdef __CUDACC__
  __host__ __device__
  #endif
  float value() const{
    if (v + w + t + q + r == 0) {
      return zero_value;
    } else {
      return (2 * v + w + 0.1 * t) / (v + w + t + 5 * q + r);
    }
  }
};

} // namespace lincs

template<>
__inline__
lincs::Desirability* Host::alloc<lincs::Desirability>(const std::size_t n) {
  return Host::force_alloc<lincs::Desirability>(n);
}

template<>
__inline__
void Host::memset<lincs::Desirability>(const std::size_t n, const char v, lincs::Desirability* const p) {
  Host::force_memset<lincs::Desirability>(n, v, p);
}

template<>
__inline__
lincs::Desirability* Device::alloc<lincs::Desirability>(const std::size_t n) {
  return Device::force_alloc<lincs::Desirability>(n);
}

template<>
__inline__
void Device::memset<lincs::Desirability>(const std::size_t n, const char v, lincs::Desirability* const p) {
  Device::force_memset<lincs::Desirability>(n, v, p);
}

#endif  // LINCS__LEARNING__WEIGHTS_PROFILES_BREED_MRSORT__IMPROVE_PROFILES__ACCURACY_HEURISTIC__DESIRABILITY_HPP
