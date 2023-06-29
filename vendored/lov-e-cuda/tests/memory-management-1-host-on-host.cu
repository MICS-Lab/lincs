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
NonTrivialButAllocable* Host::alloc<NonTrivialButAllocable>(const std::size_t n) {
  return Host::force_alloc<NonTrivialButAllocable>(n);
}

template<>
void Host::memset<NonTrivialButAllocable>(const std::size_t n, const char v, NonTrivialButAllocable* const p) {
  Host::force_memset<NonTrivialButAllocable>(n, v, p);
}

TEST(HostAllocOnHostTest, AllocateNonZero) {
  int* const p = Host::alloc<int>(10);
  EXPECT_NE(p, nullptr);
  Host::free(p);
}

TEST(HostAllocOnHostTest, AllocateZero) {
  int* const p = Host::alloc<int>(0);
  EXPECT_EQ(p, nullptr);
  Host::free(p);
}

TEST(HostAllocOnHostTest, AllocateZeroed) {
  int* const p = Host::alloc_zeroed<int>(10);
  EXPECT_NE(p, nullptr);
  EXPECT_EQ(p[0], 0);
  EXPECT_EQ(p[9], 0);
  Host::free(p);
}

TEST(HostAllocOnHostTest, AllocateTrivialAndNonTrivial) {
  // Can allocate any trivial type (https://en.cppreference.com/w/cpp/named_req/TrivialType)
  Host::free(Host::alloc<float>(1));
  Host::free(Host::alloc<double>(1));
  Host::free(Host::alloc<Trivial>(1));

  // Can't allocate non-trivial types
  #if EXPECT_COMPILE_ERROR == __LINE__
    Host::alloc<NonTrivial>(1);
  #endif

  // Can allocate non-trivial type if you insist
  Host::free(Host::alloc<NonTrivialButAllocable>(1));
  Host::free(Host::alloc_zeroed<NonTrivialButAllocable>(1));

  // But force_alloc is not directly available to clients
  #if EXPECT_COMPILE_ERROR == __LINE__
    Host::force_alloc<NonTrivial>(1);
  #endif
}

class HostMemsetOnHostTest : public testing::Test {
 protected:
  void SetUp() override {
    mem = Host::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Host::free(mem);
  }

  uint16_t* mem;
  const unsigned count = 16;
};

TEST_F(HostMemsetOnHostTest, Memset) {
  Host::memset(count, 0xAA, mem);
  EXPECT_EQ(mem[0], 0xAAAA);
  EXPECT_EQ(mem[count - 1], 0xAAAA);
}

TEST_F(HostMemsetOnHostTest, Memreset) {
  mem[0] = 0xABCD;
  mem[count - 1] = 0x1234;
  Host::memreset(count, mem);
  EXPECT_EQ(mem[0], 0);
  EXPECT_EQ(mem[count - 1], 0);
}

TEST_F(HostMemsetOnHostTest, MemsetTrivialAndNonTrivial) {
  // Can memset any trivial type (https://en.cppreference.com/w/cpp/named_req/TrivialType)
  {
    auto p = Host::alloc<float>(10);
    Host::memset(10, 0x77, p);
    EXPECT_FLOAT_EQ(p[0], 5.01922e+33);
    EXPECT_FLOAT_EQ(p[9], 5.01922e+33);
    Host::free(p);
  }

  // Can't memset non-trivial types
  {
    #if EXPECT_COMPILE_ERROR == __LINE__
      NonTrivial* p = nullptr;
      Host::memset(0, 0x77, p);
    #endif
  }

  // Can memset non-trivial type if you insist
  {
    auto p = Host::alloc<NonTrivialButAllocable>(1);
    Host::memset(1, 0x77, p);
    Host::free(p);
  }

  // But force_memset is not directly available to clients
  {
    #if EXPECT_COMPILE_ERROR == __LINE__
      NonTrivial* p = nullptr;
      Host::force_memset(1, 0x77, p);
    #endif
  }
}

class CopyOnHostTest : public testing::Test {
 protected:
  void SetUp() override {
    h1 = Host::alloc<uint16_t>(count);
    h2 = Host::alloc<uint16_t>(count);
    d1 = Device::alloc<uint16_t>(count);
    d2 = Device::alloc<uint16_t>(count);
  }

  void TearDown() override {
    Host::free(h1);
    Host::free(h2);
    Device::free(d1);
    Device::free(d2);
  }

  uint16_t* h1;
  uint16_t* h2;
  uint16_t* d1;
  uint16_t* d2;
  const std::size_t count = 16;
};

TEST_F(CopyOnHostTest, CopyHostToHost) {
  h1[0] = 42;
  h1[count - 1] = 65;
  From<Host>::To<Host>::copy(count, h1, h2);
  EXPECT_EQ(h2[0], 42);
  EXPECT_EQ(h1[count - 1], 65);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_CopyOnHostTest_CopyHostToDevice(const std::size_t count, const uint16_t* const d1) {
  assert(d1[0] == 42);
  assert(d1[count - 1] == 65);
}

TEST_F(CopyOnHostTest, CopyHostToDevice) {
  h1[0] = 42;
  h1[count - 1] = 65;
  From<Host>::To<Device>::copy(count, h1, d1);
  kernel_CopyOnHostTest_CopyHostToDevice<<<1, 1>>>(count, d1);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_CopyOnHostTest_CopyDeviceToHost(const std::size_t count, uint16_t* const d1) {
  d1[0] = 42;
  d1[count - 1] = 65;
}

TEST_F(CopyOnHostTest, CopyDeviceToHost) {
  kernel_CopyOnHostTest_CopyDeviceToHost<<<1, 1>>>(count, d1);
  check_last_cuda_error_sync_device();
  From<Device>::To<Host>::copy(count, d1, h1);
  EXPECT_EQ(h1[0], 42);
  EXPECT_EQ(h1[count - 1], 65);
}

__global__ void kernel_CopyOnHostTest_CopyDeviceToDevice(
  const std::size_t count, uint16_t* const d1, uint16_t* const d2
) {
  // cudaMemcpy is __host__ only
  #if EXPECT_COMPILE_ERROR == __LINE__
    From<Device>::To<Device>::copy(count, d1, d2);
  #endif
}

TEST(CloneTest, CloneHostToHost) {
  const std::size_t count = 16;
  uint16_t* h1 = Host::alloc<uint16_t>(count);
  h1[0] = 42;
  h1[count - 1] = 65;
  uint16_t* h2 = From<Host>::To<Host>::clone(count, h1);
  EXPECT_EQ(h2[0], 42);
  EXPECT_EQ(h2[count - 1], 65);

  Host::free(h1);
  Host::free(h2);
}

TEST(CloneTest, CloneHostToDevice) {
  const std::size_t count = 16;
  uint16_t* h = Host::alloc<uint16_t>(count);
  h[0] = 42;
  h[count - 1] = 65;
  uint16_t* d = From<Host>::To<Device>::clone(count, h);
  kernel_CopyOnHostTest_CopyHostToDevice<<<1, 1>>>(count, d);
  check_last_cuda_error_sync_device();

  Device::free(d);
  Host::free(h);
}

TEST(CloneTest, CloneDeviceToHost) {
  const std::size_t count = 16;
  uint16_t* d = Device::alloc<uint16_t>(count);
  kernel_CopyOnHostTest_CopyDeviceToHost<<<1, 1>>>(count, d);
  uint16_t* h = From<Device>::To<Host>::clone(count, d);
  EXPECT_EQ(h[0], 42);
  EXPECT_EQ(h[count - 1], 65);

  Host::free(h);
  printf("B: d=%p\n", d);
  Device::free(d);
}
