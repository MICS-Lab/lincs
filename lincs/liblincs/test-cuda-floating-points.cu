// Copyright 2023-2024 Vincent Jacques

#include <random>
#include <iostream>
#include <iomanip>

#include "vendored/lov-e.hpp"

#include "vendored/doctest.h"  // Keep last because it defines really common names like CHECK that we don't want injected into other headers


namespace {

bool env_is_true(const char* name) {
  const char* value = std::getenv(name);
  return value && std::string(value) == "true";
}

const bool forbid_gpu = env_is_true("LINCS_DEV_FORBID_GPU");

}  // namespace

namespace {

typedef GridFactory1D<128> grid;

__global__ void kernel(
  const ArrayView2D<Device, const double> operands,
  const ArrayView2D<Device, double> results
) {
  assert(results.s0() == operands.s0());
  assert(results.s1() == 4);

  unsigned index = grid::x();
  assert(index < operands.s0() + grid::blockDim().x);
  if (index < operands.s0()) {
    results[0][index] = operands[0][index] + operands[1][index];
    results[1][index] = operands[0][index] - operands[1][index];
    results[2][index] = operands[0][index] * operands[1][index];
    results[3][index] = operands[0][index] / operands[1][index];
  }
}

TEST_CASE("floating point operations on random doubles behave bit-to-bit-the-same on the GPU and on the CPU" * doctest::skip(forbid_gpu)) {
  const unsigned count = 1'000'000;
  for (unsigned random_seed = 0; random_seed != 25; ++random_seed) {
    std::mt19937 gen(random_seed);
    std::uniform_real_distribution<double> dist(0, 1);

    Array2D<Host, double> operands_on_host(2, count, uninitialized);
    for (unsigned i = 0; i < count; ++i) {
      operands_on_host[0][i] = dist(gen);
      operands_on_host[1][i] = dist(gen);
    }
    Array2D<Device, double> operands_on_device = operands_on_host.clone_to<Device>();

    Array2D<Host, double> host_results(4, count, uninitialized);
    for (unsigned index = 0; index < count; ++index) {
      host_results[0][index] = operands_on_host[0][index] + operands_on_host[1][index];
      host_results[1][index] = operands_on_host[0][index] - operands_on_host[1][index];
      host_results[2][index] = operands_on_host[0][index] * operands_on_host[1][index];
      host_results[3][index] = operands_on_host[0][index] / operands_on_host[1][index];
    }

    Array2D<Device, double> device_results(4, count, uninitialized);
    Grid grid = grid::make(count);
    kernel<<<LOVE_CONFIG(grid)>>>(operands_on_device, ref(device_results));
    Array2D<Host, double> device_results_on_host = device_results.clone_to<Host>();

    for (unsigned index = 0; index < count; ++index) {
      CHECK(device_results_on_host[0][index] == host_results[0][index]);
      CHECK(device_results_on_host[1][index] == host_results[1][index]);
      CHECK(device_results_on_host[2][index] == host_results[2][index]);
      CHECK(device_results_on_host[3][index] == host_results[3][index]);
    }
  }
}

}  // namespace


namespace {

__host__ __device__
double compute(double a, double b, double c, double d) {
  return (c * a - b * d) / a;
}

__global__ void kernel(
  const ArrayView1D<Device, const double> inputs,
  const ArrayView1D<Device, double> outputs
) {
  assert(inputs.s0() == 4);

  outputs[0] = compute(inputs[0], inputs[1], inputs[2], inputs[3]);
}

TEST_CASE("Specific operations on specific values behave differently on the GPU and on the CPU" * doctest::skip(forbid_gpu)) {
  Array1D<Host, double> inputs_on_host(4, uninitialized);
  inputs_on_host[0] = 0.6636555388921431;
  inputs_on_host[1] = 2.3023796233172344;
  inputs_on_host[2] = -5.234825899201804;
  inputs_on_host[3] = 0.004927249084663451;
  Array1D<Device, double> inputs_on_device = inputs_on_host.clone_to<Device>();

  const double expected_result = -5.251919703482374;
  const double unexpected_result = -5.251919703482373;
  assert(expected_result != unexpected_result);

  Array1D<Host, double> host_outputs(1, uninitialized);
  host_outputs[0] = compute(inputs_on_host[0], inputs_on_host[1], inputs_on_host[2], inputs_on_host[3]);
  CHECK(host_outputs[0] == expected_result);

  Array1D<Device, double> device_outputs(1, uninitialized);
  kernel<<<1, 1>>>(inputs_on_device, ref(device_outputs));
  Array1D<Host, double> device_outputs_on_host = device_outputs.clone_to<Host>();
  CHECK(device_outputs_on_host[0] != expected_result);
  CHECK(device_outputs_on_host[0] == unexpected_result);
}

}
