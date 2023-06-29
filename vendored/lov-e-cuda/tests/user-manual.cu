// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


template<typename T>
__host__ __device__
void unused(T) {}

__device__ void f(float) {}

// BEGIN BadMultiDimArray-decl
__global__ void kernel(float* data, const unsigned width, const unsigned height)
// END BadMultiDimArray-decl
{
  unsigned x = 0;
  unsigned y = 0;

  f(
  // BEGIN BadMultiDimArray-use
  data[y * width + x]
  // END BadMultiDimArray-use
  );
}

TEST(UserManual, BadMultiDimArray) {
  const unsigned width = 12;
  const unsigned height = 24;

  // BEGIN BadMultiDimArray-alloc
  float* data;
  cudaMalloc(&data, width * height * sizeof(float));
  // END BadMultiDimArray-alloc

  // BEGIN BadMultiDimArray-call
  kernel<<<1, 1>>>(data, width, height);
  check_last_cuda_error_sync_device();
  // END BadMultiDimArray-call

  // BEGIN BadMultiDimArray-free
  cudaFree(data);
  // END BadMultiDimArray-free
}

// BEGIN GoodMultiDimArray-decl
__global__ void kernel(ArrayView2D<Device, float> data)
// END GoodMultiDimArray-decl
{
  unsigned x = 0;
  unsigned y = 0;

  f(
  // BEGIN GoodMultiDimArray-use
  data[y][x]
  // END GoodMultiDimArray-use
  );
}

TEST(UserManual, GoodMultiDimArray) {
  const unsigned width = 12;
  const unsigned height = 24;

  // BEGIN GoodMultiDimArray-alloc
  Array2D<Device, float> data(height, width, uninitialized);
  // END GoodMultiDimArray-alloc

  // BEGIN GoodMultiDimArray-call
  kernel<<<1, 1>>>(ref(data));
  check_last_cuda_error_sync_device();
  // END GoodMultiDimArray-call
}

// BEGIN HostArray-function-decl
void f(ArrayView1D<Host, int> a)
// END HostArray-function-decl
{
  unused(a);
}

TEST(UserManual, HostArray) {
  // BEGIN HostArray-decl
  Array3D<Host, int> a(42, 36, 57, zeroed);
  // END HostArray-decl

  // BEGIN HostArray-zeroes
  EXPECT_EQ(a[0][0][0], 0);
  EXPECT_EQ(a[41][35][56], 0);
  // END HostArray-zeroes

  // BEGIN HostArray-sizes
  EXPECT_EQ(a.s2(), 42);
  EXPECT_EQ(a.s1(), 36);
  EXPECT_EQ(a.s0(), 57);
  // END HostArray-sizes

  // BEGIN HostArray-indexed-sizes
  EXPECT_EQ(a[0].s1(), 36);
  EXPECT_EQ(a[0].s0(), 57);
  EXPECT_EQ(a[0][0].s0(), 57);
  // END HostArray-indexed-sizes

  // BEGIN HostArray-indexed-call
  f(a[0][12]);
  // END HostArray-indexed-call

  // BEGIN HostArray-clone
  Array3D<Device, int> b = a.clone_to<Device>();
  // END HostArray-clone

  // BEGIN HostArray-copy
  copy<Host, Device>(a, ref(b));
  // END HostArray-copy
}

// BEGIN NonTrivial-decl
struct NonTrivial {
  NonTrivial() {}
};
// END NonTrivial-decl

// BEGIN NonTrivial-special
template<>
NonTrivial* Host::alloc<NonTrivial>(const std::size_t n) {
  return Host::force_alloc<NonTrivial>(n);
}

template<>
void Host::memset<NonTrivial>(const std::size_t n, const char v, NonTrivial* const p) {
  Host::force_memset<NonTrivial>(n, v, p);
}

template<>
NonTrivial* Device::alloc<NonTrivial>(const std::size_t n) {
  return Device::force_alloc<NonTrivial>(n);
}

template<>
void Device::memset<NonTrivial>(const std::size_t n, const char v, NonTrivial* const p) {
  Device::force_memset<NonTrivial>(n, v, p);
}
// END NonTrivial-special

TEST(UserManual, NonTrivial) {
  // BEGIN NonTrivial-use
  Array1D<Host, NonTrivial> h(10, zeroed);
  Array1D<Device, NonTrivial> d(10, zeroed);
  // END NonTrivial-use
}
