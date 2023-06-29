// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


TEST(ArrayCreationTest, HostOnHost) {
  Array1D<Host, int> u1(10, uninitialized);
  EXPECT_EQ(u1.s0(), 10);

  Array2D<Host, int> u2(4, 5, uninitialized);
  EXPECT_EQ(u2.s0(), 5);
  EXPECT_EQ(u2.s1(), 4);
  EXPECT_EQ(u2[0].s0(), 5);

  Array5D<Host, int> u5(7, 6, 3, 4, 8, uninitialized);
  EXPECT_EQ(u5.s0(), 8);
  EXPECT_EQ(u5.s1(), 4);
  EXPECT_EQ(u5.s2(), 3);
  EXPECT_EQ(u5.s3(), 6);
  EXPECT_EQ(u5.s4(), 7);
  EXPECT_EQ(u5[0][0][0][0].s0(), 8);

  Array1D<Host, int> z1(1, zeroed);
  EXPECT_EQ(z1.s0(), 1);
  EXPECT_EQ(z1[0], 0);

  Array2D<Host, int> z2(1, 1, zeroed);
  EXPECT_EQ(z2.s0(), 1);
  EXPECT_EQ(z2.s1(), 1);
  EXPECT_EQ(z2[0].s0(), 1);
  EXPECT_EQ(z2[0][0], 0);

  Array5D<Host, int> z5(1, 1, 1, 1, 1, zeroed);
  EXPECT_EQ(z5.s0(), 1);
  EXPECT_EQ(z5.s1(), 1);
  EXPECT_EQ(z5.s2(), 1);
  EXPECT_EQ(z5.s3(), 1);
  EXPECT_EQ(z5.s4(), 1);
  EXPECT_EQ(z5[0][0][0][0].s0(), 1);
  EXPECT_EQ(z5[0][0][0][0][0], 0);
}

__global__ void kernel_ArrayCreationTest_HostOnDevice() {
  #if EXPECT_COMPILE_ERROR == __LINE__
    Array1D<Host, int> u1(10, uninitialized);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array2D<Host, int> u2(4, 5, uninitialized);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array5D<Host, int> u5(7, 6, 3, 4, 8, uninitialized);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array1D<Host, int> z1(1, zeroed);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array2D<Host, int> z2(1, 1, zeroed);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array5D<Host, int> z5(1, 1, 1, 1, 1, zeroed);
  #endif

  // Note that destructing a Array<Host> in device code does compile
  // but fails at runtime (if not null).
  // We'd love to make this a compile-time error, but we don't know how.
  Array1D<Host, int>* p1 = nullptr;
  delete p1;
  Array2D<Host, int>* p2 = nullptr;
  delete p2;
  Array5D<Host, int>* p5 = nullptr;
  delete p5;
}

TEST(ArrayCreationTest, HostOnDevice) {
  kernel_ArrayCreationTest_HostOnDevice<<<1, 1>>>();
  check_last_cuda_error_sync_device();
}

__global__ void kernel_ArrayCreationTest_DeviceOnHost(
  ArrayView1D<Device, const int> u1,
  ArrayView2D<Device, const int> u2,
  ArrayView5D<Device, const int> u5,
  ArrayView1D<Device, const int> z1,
  ArrayView2D<Device, const int> z2,
  ArrayView5D<Device, const int> z5
) {
  assert(u1.s0() == 10);

  assert(u2.s0() == 5);
  assert(u2.s1() == 4);
  assert(u2[0].s0() == 5);

  assert(u5.s0() == 8);
  assert(u5.s1() == 4);
  assert(u5.s2() == 3);
  assert(u5.s3() == 6);
  assert(u5.s4() == 7);
  assert(u5[0][0][0][0].s0() == 8);

  assert(z1.s0() == 1);
  assert(z1[0] == 0);

  assert(z2.s0() == 1);
  assert(z2.s1() == 1);
  assert(z2[0].s0() == 1);
  assert(z2[0][0] == 0);

  assert(z5.s0() == 1);
  assert(z5.s1() == 1);
  assert(z5.s2() == 1);
  assert(z5.s3() == 1);
  assert(z5.s4() == 1);
  assert(z5[0][0][0][0].s0() == 1);
  assert(z5[0][0][0][0][0] == 0);
}

TEST(ArrayCreationTest, DeviceOnHost) {
  Array1D<Device, int> u1(10, uninitialized);
  EXPECT_EQ(u1.s0(), 10);

  Array2D<Device, int> u2(4, 5, uninitialized);
  EXPECT_EQ(u2.s0(), 5);
  EXPECT_EQ(u2.s1(), 4);
  EXPECT_EQ(u2[0].s0(), 5);

  Array5D<Device, int> u5(7, 6, 3, 4, 8, uninitialized);
  EXPECT_EQ(u5.s0(), 8);
  EXPECT_EQ(u5.s1(), 4);
  EXPECT_EQ(u5.s2(), 3);
  EXPECT_EQ(u5.s3(), 6);
  EXPECT_EQ(u5.s4(), 7);
  EXPECT_EQ(u5[0][0][0][0].s0(), 8);

  Array1D<Device, int> z1(1, zeroed);
  EXPECT_EQ(z1.s0(), 1);
  // @todo #if EXPECT_COMPILE_ERROR == __LINE__
  //   z1[0];
  // #endif

  Array2D<Device, int> z2(1, 1, zeroed);
  EXPECT_EQ(z2.s0(), 1);
  EXPECT_EQ(z2.s1(), 1);
  EXPECT_EQ(z2[0].s0(), 1);
  #if EXPECT_COMPILE_ERROR == __LINE__
    z2[0][0];
  #endif

  Array5D<Device, int> z5(1, 1, 1, 1, 1, zeroed);
  EXPECT_EQ(z5.s0(), 1);
  EXPECT_EQ(z5.s1(), 1);
  EXPECT_EQ(z5.s2(), 1);
  EXPECT_EQ(z5.s3(), 1);
  EXPECT_EQ(z5.s4(), 1);
  EXPECT_EQ(z5[0][0][0][0].s0(), 1);
  #if EXPECT_COMPILE_ERROR == __LINE__
    z5[0][0][0][0][0];
  #endif

  kernel_ArrayCreationTest_DeviceOnHost<<<1, 1>>>(u1, u2, u5, z1, z2, z5);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_ArrayCreationTest_DeviceOnDevice() {
  Array1D<Device, int> u1(10, uninitialized);
  assert(u1.s0() == 10);

  Array2D<Device, int> u2(4, 5, uninitialized);
  assert(u2.s0() == 5);
  assert(u2.s1() == 4);
  assert(u2[0].s0() == 5);

  Array5D<Device, int> u5(7, 6, 3, 4, 8, uninitialized);
  assert(u5.s0() == 8);
  assert(u5.s1() == 4);
  assert(u5.s2() == 3);
  assert(u5.s3() == 6);
  assert(u5.s4() == 7);
  assert(u5[0][0][0][0].s0() == 8);

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array1D<Device, int> z1(1, zeroed);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array2D<Device, int> z2(1, 1, zeroed);
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    Array5D<Device, int> z5(1, 1, 1, 1, 1, zeroed);
  #endif
}

TEST(ArrayCreationTest, DeviceOnDevice) {
  kernel_ArrayCreationTest_DeviceOnDevice<<<1, 1>>>();
  check_last_cuda_error_sync_device();
}
