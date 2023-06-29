// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


TEST(HostArrayOnHostTest, AllocateZeroed) {
  Array2D<Host, int> a(3, 4, zeroed);
  EXPECT_EQ(a[0][0], 0);
}

TEST(HostArrayOnHostTest, Assign) {
  Array2D<Host, int> a1(3, 4, uninitialized);
  const int* data = a1.data();
  Array2D<Host, int> a2(1, 1, uninitialized);

  // Movable
  a2 = std::move(a1);

  // Not copyable
  #if EXPECT_COMPILE_ERROR == __LINE__
    a2 = a1;
  #endif

  // 'a1' has been emptied
  EXPECT_EQ(a1.s1(), 0);
  EXPECT_EQ(a1.s0(), 0);
  EXPECT_EQ(a1.data(), nullptr);

  // 'a2' now owns the data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_EQ(a2.data(), data);
}

TEST(HostArrayOnHostTest, MoveConstruct) {
  Array2D<Host, int> a1(3, 4, uninitialized);
  const int* data = a1.data();

  // Movable
  Array2D<Host, int> a2(std::move(a1));

  // Not copyable
  #if EXPECT_COMPILE_ERROR == __LINE__
    Array2D<Host, int> a3(a1);
  #endif

  // 'a1' has been emptied
  EXPECT_EQ(a1.s1(), 0);
  EXPECT_EQ(a1.s0(), 0);
  EXPECT_EQ(a1.data(), nullptr);

  // 'a2' now owns the data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_EQ(a2.data(), data);
}

TEST(HostArrayOnHostTest, CloneToHost) {
  Array2D<Host, int> a1(3, 4, uninitialized);
  a1[0][0] = 42;
  a1[2][3] = 65;

  Array2D<Host, int> a2 = a1.clone_to<Host>();

  // 'a1' and 'a2' each own their own data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_NE(a2.data(), a1.data());
  EXPECT_EQ(a2[0][0], 42);
  EXPECT_EQ(a2[2][3], 65);
}

__global__ void kernel_HostArrayOnHostTest_CloneToHost(ArrayView2D<Device, const int> a2) {
  assert(a2[0][0] == 42);
  assert(a2[2][3] == 65);
}

TEST(HostArrayOnHostTest, CloneToDevice) {
  Array2D<Host, int> a1(3, 4, uninitialized);
  a1[0][0] = 42;
  a1[2][3] = 65;

  Array2D<Device, int> a2 = a1.clone_to<Device>();

  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  kernel_HostArrayOnHostTest_CloneToHost<<<1, 1>>>(a2);
  check_last_cuda_error_sync_device();
}

TEST(DeviceArrayOnHostTest, Assign) {
  Array2D<Device, int> a1(3, 4, uninitialized);
  const int* data = a1.data();
  Array2D<Device, int> a2(1, 1, uninitialized);

  // Movable
  a2 = std::move(a1);

  // Not copyable
  #if EXPECT_COMPILE_ERROR == __LINE__
    a2 = a1;
  #endif

  // 'a1' has been emptied
  EXPECT_EQ(a1.s1(), 0);
  EXPECT_EQ(a1.s0(), 0);
  EXPECT_EQ(a1.data(), nullptr);

  // 'a2' now owns the data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_EQ(a2.data(), data);
}

TEST(DeviceArrayOnHostTest, MoveConstruct) {
  Array2D<Device, int> a1(3, 4, uninitialized);
  const int* data = a1.data();

  // Movable
  Array2D<Device, int> a2(std::move(a1));

  // Not copyable
  #if EXPECT_COMPILE_ERROR == __LINE__
    Array2D<Device, int> a3(a1);
  #endif

  // 'a1' has been emptied
  EXPECT_EQ(a1.s1(), 0);
  EXPECT_EQ(a1.s0(), 0);
  EXPECT_EQ(a1.data(), nullptr);

  // 'a2' now owns the data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_EQ(a2.data(), data);
}

__global__ void kernel_DeviceArrayOnHostTest_CloneToHost(ArrayView2D<Device, int> a1) {
  a1[0][0] = 42;
  a1[2][3] = 65;
}

TEST(DeviceArrayOnHostTest, CloneToHost) {
  Array2D<Device, int> a1(3, 4, uninitialized);
  kernel_DeviceArrayOnHostTest_CloneToHost<<<1, 1>>>(ref(a1));
  check_last_cuda_error_sync_device();

  Array2D<Host, int> a2 = a1.clone_to<Host>();

  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_EQ(a2[0][0], 42);
  EXPECT_EQ(a2[2][3], 65);
}

__global__ void kernel_DeviceArrayOnHostTest_CloneToDevice_1(ArrayView2D<Device, int> a1) {
  a1[0][0] = 42;
  a1[2][3] = 65;
}

__global__ void kernel_HostArrayOnHostTest_CloneToDevice_2(ArrayView2D<Device, const int> a2) {
  assert(a2[0][0] == 42);
  assert(a2[2][3] == 65);
}

TEST(DeviceArrayOnHostTest, CloneToDevice) {
  Array2D<Device, int> a1(3, 4, uninitialized);
  kernel_DeviceArrayOnHostTest_CloneToDevice_1<<<1, 1>>>(ref(a1));
  check_last_cuda_error_sync_device();

  Array2D<Device, int> a2 = a1.clone_to<Device>();

  // 'a1' and 'a2' each own their own data
  EXPECT_EQ(a2.s1(), 3);
  EXPECT_EQ(a2.s0(), 4);
  EXPECT_NE(a2.data(), a1.data());
  kernel_HostArrayOnHostTest_CloneToDevice_2<<<1, 1>>>(a2);
}

__global__ void kernel_DeviceArrayOnDeviceTest_Create() {
  Array1D<Device, int> a1(10, uninitialized);
  Array3D<Device, int> a3(10, 10, 10, uninitialized);
}

TEST(DeviceArrayOnDeviceTest, Create) {
  kernel_DeviceArrayOnDeviceTest_Create<<<1, 1>>>();
  check_last_cuda_error_sync_device();
}
