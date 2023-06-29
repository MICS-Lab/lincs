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

TEST(DeviceAllocOnHostTest, AllocateNonZero) {
  int* const p = Device::alloc<int>(10);
  EXPECT_NE(p, nullptr);
  Device::free(p);
}

TEST(DeviceAllocOnHostTest, AllocateZero) {
  int* const p = Device::alloc<int>(0);
  EXPECT_EQ(p, nullptr);
  Device::free(p);
}

__global__ void kernel_DeviceAllocOnHostTest_AllocateZeroed(int* p) {
  assert(p[0] == 0);
  assert(p[9] == 0);
}

TEST(DeviceAllocOnHostTest, AllocateZeroed) {
  int* const p = Device::alloc_zeroed<int>(10);
  EXPECT_NE(p, nullptr);
  kernel_DeviceAllocOnHostTest_AllocateZeroed<<<1, 1>>>(p);
  check_last_cuda_error_sync_device();
  Device::free(p);
}

TEST(DeviceAllocOnHostTest, AllocateTrivialAndNonTrivial) {
  // Can allocate any trivial type
  Device::free(Device::alloc<float>(1));
  Device::free(Device::alloc<double>(1));
  Device::free(Device::alloc<Trivial>(1));

  // Can't allocate non-trivial types
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::alloc<NonTrivial>(1);
  #endif

  // Can allocate non-trivial type if you insist
  Device::free(Device::alloc<NonTrivialButAllocable>(1));
  Device::free(Device::alloc_zeroed<NonTrivialButAllocable>(1));

  // But force_alloc is not directly available to clients
  #if EXPECT_COMPILE_ERROR == __LINE__
    Device::force_alloc<float>(1);
  #endif
}

class DeviceMemsetOnHostTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = Device::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Device::free(mem);
  }

  uint16_t* mem;
  const std::size_t count = 16;
};

__global__ void kernel_DeviceMemsetOnHostTest_Memset(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0xAAAA);
  assert(mem[count - 1] == 0xAAAA);
}

TEST_F(DeviceMemsetOnHostTest, Memset) {
  Device::memset(count, 0xAA, mem);
  kernel_DeviceMemsetOnHostTest_Memset<<<1, 1>>>(count, mem);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_DeviceMemsetOnHostTest_Memreset_1(const std::size_t count, uint16_t* const mem) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
}

__global__ void kernel_DeviceMemsetOnHostTest_Memreset_2(const std::size_t count, const uint16_t* const mem) {
  assert(mem[0] == 0);
  assert(mem[count - 1] == 0);
}

TEST_F(DeviceMemsetOnHostTest, Memreset) {
  kernel_DeviceMemsetOnHostTest_Memreset_1<<<1, 1>>>(count, mem);
  Device::memreset(count, mem);
  kernel_DeviceMemsetOnHostTest_Memreset_2<<<1, 1>>>(count, mem);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_DeviceMemsetOnHostTest_Memreset_2(const float* const p) {
  assert(std::abs(p[0] - 5.01922e+33) < 1e27);
  assert(std::abs(p[9] - 5.01922e+33) < 1e27);
}

TEST_F(DeviceMemsetOnHostTest, MemsetTrivialAndNonTrivial) {
  // Can memset any trivial type (https://en.cppreference.com/w/cpp/named_req/TrivialType)
  {
    auto p = Device::alloc<float>(10);
    Device::memset(10, 0x77, p);
    kernel_DeviceMemsetOnHostTest_Memreset_2<<<1, 1>>>(p);
    Device::free(p);
  }

  // Can't memset non-trivial types
  {
    #if EXPECT_COMPILE_ERROR == __LINE__
      NonTrivial* p = nullptr;
      Device::memset(0, 0x77, p);
    #endif
  }

  // Can memset non-trivial type if you insist
  {
    auto p = Device::alloc<NonTrivialButAllocable>(1);
    Device::memset(1, 0x77, p);
    Device::free(p);
  }

  // But force_memset is not directly available to clients
  {
    #if EXPECT_COMPILE_ERROR == __LINE__
      NonTrivial* p = nullptr;
      Device::force_memset(1, 0x77, p);
    #endif
  }
}
