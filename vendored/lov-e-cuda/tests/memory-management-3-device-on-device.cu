// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


struct Trivial {
  int i;
  char c;
  bool b;
  float f;
  double d;
};

struct NonTrivial {
  NonTrivial() {}
};

struct NonTrivialButAllocable {
  NonTrivialButAllocable() {}
};

template<>
NonTrivialButAllocable* Device::alloc<NonTrivialButAllocable>(const std::size_t n) {
  return Device::force_alloc<NonTrivialButAllocable>(n);
}

template<>
void Device::memset<NonTrivialButAllocable>(const std::size_t n, const char v, NonTrivialButAllocable* const p) {
  Device::force_memset<NonTrivialButAllocable>(n, v, p);
}

__global__ void kernel_DeviceAllocOnDeviceTest_AllocateNonZero() {
  int* const p = Device::alloc<int>(10);
  assert(p != nullptr);
  Device::free(p);
}

TEST(DeviceAllocOnDeviceTest, AllocateNonZero) {
  kernel_DeviceAllocOnDeviceTest_AllocateNonZero<<<1, 1>>>();
  check_last_cuda_error_sync_device();
}

__global__ void kernel_DeviceAllocOnDeviceTest_AllocateZero() {
  int* const p = Device::alloc<int>(0);
  assert(p == nullptr);
  Device::free(p);
}

TEST(DeviceAllocOnDeviceTest, AllocateZero) {
  kernel_DeviceAllocOnDeviceTest_AllocateZero<<<1, 1>>>();
  check_last_cuda_error_sync_device();
}

__global__ void kernel_UnavalaibleOnDevice() {
  // cudaMemset is __host__ only, so anything that ends up calling it is __host__ only
  #if EXPECT_COMPILE_ERROR == __LINE__
    int* const p = Device::alloc_zeroed<int>(10);
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::memset(0, 0, reinterpret_cast<float*>(nullptr));
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::memreset(0, reinterpret_cast<float*>(nullptr));
  #endif
}
