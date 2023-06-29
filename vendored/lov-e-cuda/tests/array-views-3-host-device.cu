// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


__host__
int host_function(ArrayView1D<Device, int> a) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    return a[0];
  #else
    return a.s0();
  #endif
}

__device__
int device_function(ArrayView1D<Host, int> a) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    return a[0];
  #else
    return a.s0();
  #endif
}

__host__
int host_function(ArrayView3D<Device, int> a) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    return a[0][0][0];
  #else
    return a.s0();
  #endif
}

__device__
int device_function(ArrayView3D<Host, int> a) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    return a[0][0][0];
  #else
    return a.s0();
  #endif
}

__host__ __device__
int host_device_function_1d(ArrayView1D<Anywhere, int> a) {
  return a[0];
}

__global__ void kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction1D(ArrayView1D<Device, int> a) {
  assert(host_device_function_1d(a) == 0);
}

TEST(ArrayViewHostDeviceTest, PassToHostDeviceFunction1D) {
  Array1D<Host, int> h_a(1, zeroed);
  EXPECT_EQ(host_device_function_1d(ref(h_a)), 0);

  ArrayView1D<Host, int> h_av(ref(h_a));
  EXPECT_EQ(host_device_function_1d(h_av), 0);

  Array1D<Device, int> d_a(1, zeroed);
  kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction1D<<<1, 1>>>(ref(d_a));
  check_last_cuda_error_sync_device();
}

__host__ __device__
int host_device_function_1d_const(ArrayView1D<Anywhere, const int> a) {
  return a[0];
}

__global__ void kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction1DConst(ArrayView1D<Device, int> a) {
  assert(host_device_function_1d_const(a) == 0);
}

TEST(ArrayViewHostDeviceTest, PassToHostDeviceFunction1DConst) {
  Array1D<Host, int> h_a(1, zeroed);
  EXPECT_EQ(host_device_function_1d_const(h_a), 0);

  ArrayView1D<Host, int> h_av(ref(h_a));
  EXPECT_EQ(host_device_function_1d_const(h_av), 0);

  Array1D<Device, int> d_a(1, zeroed);
  kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction1DConst<<<1, 1>>>(ref(d_a));
  check_last_cuda_error_sync_device();
}

__host__ __device__
int host_device_function_3d(ArrayView3D<Anywhere, int> a) {
  return a[0][0][0];
}

__global__ void kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction3D(ArrayView3D<Device, int> a) {
  assert(host_device_function_3d(a) == 0);
}

TEST(ArrayViewHostDeviceTest, PassToHostDeviceFunction3D) {
  Array3D<Host, int> h_a(1, 1, 1, zeroed);
  EXPECT_EQ(host_device_function_3d(ref(h_a)), 0);

  ArrayView3D<Host, int> h_av(ref(h_a));
  EXPECT_EQ(host_device_function_3d(h_av), 0);

  Array3D<Device, int> d_a(1, 1, 1, zeroed);
  kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction3D<<<1, 1>>>(ref(d_a));
  check_last_cuda_error_sync_device();
}

__host__ __device__
int host_device_function_3d_const(ArrayView3D<Anywhere, const int> a) {
  return a[0][0][0];
}

__global__ void kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction3DConst(ArrayView3D<Device, int> a) {
  assert(host_device_function_3d_const(a) == 0);
}

TEST(ArrayViewHostDeviceTest, PassToHostDeviceFunction3DConst) {
  Array3D<Host, int> h_a(1, 1, 1, zeroed);
  EXPECT_EQ(host_device_function_3d_const(h_a), 0);

  ArrayView3D<Host, int> h_av(ref(h_a));
  EXPECT_EQ(host_device_function_3d_const(h_av), 0);

  Array3D<Device, int> d_a(1, 1, 1, zeroed);
  kernel_ArrayViewHostDeviceTest_PassToHostDeviceFunction3DConst<<<1, 1>>>(ref(d_a));
  check_last_cuda_error_sync_device();
}
