// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


struct CopyTest : testing::Test {
  CopyTest() :
    h1(1, zeroed), h2(2, 1, zeroed), h5(5, 4, 3, 2, 1, zeroed),
    d1(1, zeroed), d2(2, 1, zeroed), d5(5, 4, 3, 2, 1, zeroed)
  {}

  Array1D<Host, int> h1;
  Array2D<Host, int> h2;
  Array5D<Host, int> h5;
  Array1D<Device, int> d1;
  Array2D<Device, int> d2;
  Array5D<Device, int> d5;
};

TEST_F(CopyTest, HostToHostOnHost) {
  Array1D<Host, int> c1(1, uninitialized);
  copy(h1, ref(c1));
  EXPECT_EQ(c1[0], 0);

  Array2D<Host, int> c2(2, 1, uninitialized);
  copy(h2, ref(c2));
  EXPECT_EQ(c2[0][0], 0);

  Array5D<Host, int> c5(5, 4, 3, 2, 1, uninitialized);
  copy(h5, ref(c5));
  EXPECT_EQ(c5[0][0][0][0][0], 0);
}

__device__ void CopyTest_HostToHostOnDevice(
  Array1D<Host, int>& h1,  // NOLINT(runtime/references)
  Array2D<Host, int>& h2,  // NOLINT(runtime/references)
  Array5D<Host, int>& h5,  // NOLINT(runtime/references)
  Array1D<Host, int>& c1,  // NOLINT(runtime/references)
  Array2D<Host, int>& c2,  // NOLINT(runtime/references)
  Array5D<Host, int>& c5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h1, ref(c1));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h2, ref(c2));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h5, ref(c5));
  #endif
}

TEST_F(CopyTest, HostToHostOnDevice) {
}

__global__ void kernel_CopyTest_HostToDeviceOnHost(
  ArrayView1D<Device, const int> c1,
  ArrayView2D<Device, const int> c2,
  ArrayView5D<Device, const int> c5
) {
  assert(c1[0] == 0);
  assert(c2[0][0] == 0);
  assert(c5[0][0][0][0][0] == 0);
}

TEST_F(CopyTest, HostToDeviceOnHost) {
  Array1D<Device, int> c1(1, uninitialized);
  copy(h1, ref(c1));

  Array2D<Device, int> c2(2, 1, uninitialized);
  copy(h2, ref(c2));

  Array5D<Device, int> c5(5, 4, 3, 2, 1, uninitialized);
  copy(h5, ref(c5));

  kernel_CopyTest_HostToDeviceOnHost<<<1, 1>>>(c1, c2, c5);
  check_last_cuda_error_sync_device();
}

__device__ void CopyTest_HostToDeviceOnDevice(
  Array1D<Host, int>& h1,  // NOLINT(runtime/references)
  Array2D<Host, int>& h2,  // NOLINT(runtime/references)
  Array5D<Host, int>& h5,  // NOLINT(runtime/references)
  Array1D<Device, int>& c1,  // NOLINT(runtime/references)
  Array2D<Device, int>& c2,  // NOLINT(runtime/references)
  Array5D<Device, int>& c5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h1, ref(c1));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h2, ref(c2));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(h5, ref(c5));
  #endif
}

TEST_F(CopyTest, HostToDeviceOnDevice) {
}

TEST_F(CopyTest, DeviceToHostOnHost) {
  Array1D<Host, int> c1(1, uninitialized);
  copy(d1, ref(c1));
  EXPECT_EQ(c1[0], 0);

  Array2D<Host, int> c2(2, 1, uninitialized);
  copy(d2, ref(c2));
  EXPECT_EQ(c2[0][0], 0);

  Array5D<Host, int> c5(5, 4, 3, 2, 1, uninitialized);
  copy(d5, ref(c5));
  EXPECT_EQ(c5[0][0][0][0][0], 0);
}

__device__ void CopyTest_DeviceToHostOnDevice(
  Array1D<Device, int>& d1,  // NOLINT(runtime/references)
  Array2D<Device, int>& d2,  // NOLINT(runtime/references)
  Array5D<Device, int>& d5,  // NOLINT(runtime/references)
  Array1D<Host, int>& c1,  // NOLINT(runtime/references)
  Array2D<Host, int>& c2,  // NOLINT(runtime/references)
  Array5D<Host, int>& c5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d1, ref(c1));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d2, ref(c2));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d5, ref(c5));
  #endif
}

TEST_F(CopyTest, DeviceToHostOnDevice) {
}

__global__ void kernel_CopyTest_DeviceToDeviceOnHost(
  ArrayView1D<Device, const int> c1,
  ArrayView2D<Device, const int> c2,
  ArrayView5D<Device, const int> c5
) {
  assert(c1[0] == 0);
  assert(c2[0][0] == 0);
  assert(c5[0][0][0][0][0] == 0);
}

TEST_F(CopyTest, DeviceToDeviceOnHost) {
  Array1D<Device, int> c1(1, uninitialized);
  copy(d1, ref(c1));
  EXPECT_EQ(c1.s0(), 1);
  EXPECT_NE(c1.data(), d1.data());

  Array2D<Device, int> c2(2, 1, uninitialized);
  copy(d2, ref(c2));
  EXPECT_EQ(c2.s0(), 1);
  EXPECT_EQ(c2.s1(), 2);
  EXPECT_NE(c2.data(), d2.data());

  Array5D<Device, int> c5(5, 4, 3, 2, 1, uninitialized);
  copy(d5, ref(c5));
  EXPECT_EQ(c5.s0(), 1);
  EXPECT_EQ(c5.s1(), 2);
  EXPECT_EQ(c5.s2(), 3);
  EXPECT_EQ(c5.s3(), 4);
  EXPECT_EQ(c5.s4(), 5);
  EXPECT_NE(c5.data(), d5.data());

  kernel_CopyTest_DeviceToDeviceOnHost<<<1, 1>>>(c1, c2, c5);
  check_last_cuda_error_sync_device();
}

__device__ void kernel_CopyTest_DeviceToDeviceOnDevice(
  Array1D<Device, int>& d1,  // NOLINT(runtime/references)
  Array2D<Device, int>& d2,  // NOLINT(runtime/references)
  Array5D<Device, int>& d5,  // NOLINT(runtime/references)
  Array1D<Device, int>& c1,  // NOLINT(runtime/references)
  Array2D<Device, int>& c2,  // NOLINT(runtime/references)
  Array5D<Device, int>& c5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d1, ref(c1));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d2, ref(c2));
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(d5, ref(c5));
  #endif
}

TEST_F(CopyTest, DeviceToDeviceOnDevice) {
}


struct CloneTest : testing::Test {
  CloneTest() :
    h1(1, zeroed), h2(2, 1, zeroed), h5(5, 4, 3, 2, 1, zeroed),
    d1(1, zeroed), d2(2, 1, zeroed), d5(5, 4, 3, 2, 1, zeroed)
  {}

  Array1D<Host, int> h1;
  Array2D<Host, int> h2;
  Array5D<Host, int> h5;
  Array1D<Device, int> d1;
  Array2D<Device, int> d2;
  Array5D<Device, int> d5;
};

TEST_F(CloneTest, HostToHostOnHost) {
  Array1D<Host, int> c1 = h1.clone();
  EXPECT_EQ(c1.s0(), 1);
  EXPECT_EQ(c1[0], 0);
  EXPECT_NE(c1.data(), h1.data());

  Array2D<Host, int> c2 = h2.clone();
  EXPECT_EQ(c2.s0(), 1);
  EXPECT_EQ(c2.s1(), 2);
  EXPECT_EQ(c2[0][0], 0);
  EXPECT_NE(c2.data(), h2.data());

  Array5D<Host, int> c5 = h5.clone();
  EXPECT_EQ(c5.s0(), 1);
  EXPECT_EQ(c5.s1(), 2);
  EXPECT_EQ(c5.s2(), 3);
  EXPECT_EQ(c5.s3(), 4);
  EXPECT_EQ(c5.s4(), 5);
  EXPECT_EQ(c5[0][0][0][0][0], 0);
  EXPECT_NE(c5.data(), h5.data());
}

__device__ void CloneTest_HostToHostOnDevice(
  Array1D<Host, int>& h1,  // NOLINT(runtime/references)
  Array2D<Host, int>& h2,  // NOLINT(runtime/references)
  Array5D<Host, int>& h5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    h1.clone();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    h2.clone();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    h5.clone();
  #endif
}

TEST_F(CloneTest, HostToHostOnDevice) {
}

__global__ void kernel_CloneTest_HostToDeviceOnHost(
  ArrayView1D<Device, const int> c1,
  ArrayView2D<Device, const int> c2,
  ArrayView5D<Device, const int> c5
) {
  assert(c1[0] == 0);
  assert(c2[0][0] == 0);
  assert(c5[0][0][0][0][0] == 0);
}

TEST_F(CloneTest, HostToDeviceOnHost) {
  Array1D<Device, int> c1 = h1.clone_to<Device>();
  EXPECT_EQ(c1.s0(), 1);

  Array2D<Device, int> c2 = h2.clone_to<Device>();
  EXPECT_EQ(c2.s0(), 1);
  EXPECT_EQ(c2.s1(), 2);

  Array5D<Device, int> c5 = h5.clone_to<Device>();
  EXPECT_EQ(c5.s0(), 1);
  EXPECT_EQ(c5.s1(), 2);
  EXPECT_EQ(c5.s2(), 3);
  EXPECT_EQ(c5.s3(), 4);
  EXPECT_EQ(c5.s4(), 5);

  kernel_CloneTest_HostToDeviceOnHost<<<1, 1>>>(c1, c2, c5);
  check_last_cuda_error_sync_device();
}

__device__ void CloneTest_HostToDeviceOnDevice(
  Array1D<Host, int>& h1,  // NOLINT(runtime/references)
  Array2D<Host, int>& h2,  // NOLINT(runtime/references)
  Array5D<Host, int>& h5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    h1.clone_to<Device>();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    h2.clone_to<Device>();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    h5.clone_to<Device>();
  #endif
}

TEST_F(CloneTest, HostToDeviceOnDevice) {
}

TEST_F(CloneTest, DeviceToHostOnHost) {
  Array1D<Host, int> c1 = d1.clone_to<Host>();
  EXPECT_EQ(c1.s0(), 1);
  EXPECT_EQ(c1[0], 0);

  Array2D<Host, int> c2 = d2.clone_to<Host>();
  EXPECT_EQ(c2.s0(), 1);
  EXPECT_EQ(c2.s1(), 2);
  EXPECT_EQ(c2[0][0], 0);

  Array5D<Host, int> c5 = d5.clone_to<Host>();
  EXPECT_EQ(c5.s0(), 1);
  EXPECT_EQ(c5.s1(), 2);
  EXPECT_EQ(c5.s2(), 3);
  EXPECT_EQ(c5.s3(), 4);
  EXPECT_EQ(c5.s4(), 5);
  EXPECT_EQ(c5[0][0][0][0][0], 0);
}

__device__ void CloneTest_DeviceToHostOnDevice(
  Array1D<Device, int>& d1,  // NOLINT(runtime/references)
  Array2D<Device, int>& d2,  // NOLINT(runtime/references)
  Array5D<Device, int>& d5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    d1.clone_to<Host>();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    d2.clone_to<Host>();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    d5.clone_to<Host>();
  #endif
}

TEST_F(CloneTest, DeviceToHostOnDevice) {
}

__global__ void kernel_CloneTest_DeviceToDeviceOnHost(
  ArrayView1D<Device, const int> c1,
  ArrayView2D<Device, const int> c2,
  ArrayView5D<Device, const int> c5
) {
  assert(c1[0] == 0);
  assert(c2[0][0] == 0);
  assert(c5[0][0][0][0][0] == 0);
}

TEST_F(CloneTest, DeviceToDeviceOnHost) {
  Array1D<Device, int> c1 = d1.clone();
  EXPECT_EQ(c1.s0(), 1);
  EXPECT_NE(c1.data(), d1.data());

  Array2D<Device, int> c2 = d2.clone();
  EXPECT_EQ(c2.s0(), 1);
  EXPECT_EQ(c2.s1(), 2);
  EXPECT_NE(c2.data(), d2.data());

  Array5D<Device, int> c5 = d5.clone();
  EXPECT_EQ(c5.s0(), 1);
  EXPECT_EQ(c5.s1(), 2);
  EXPECT_EQ(c5.s2(), 3);
  EXPECT_EQ(c5.s3(), 4);
  EXPECT_EQ(c5.s4(), 5);
  EXPECT_NE(c5.data(), d5.data());

  kernel_CloneTest_DeviceToDeviceOnHost<<<1, 1>>>(c1, c2, c5);
  check_last_cuda_error_sync_device();
}

__device__ void kernel_CloneTest_DeviceToDeviceOnDevice(
  Array1D<Device, int>& d1,  // NOLINT(runtime/references)
  Array2D<Device, int>& d2,  // NOLINT(runtime/references)
  Array5D<Device, int>& d5  // NOLINT(runtime/references)
) {
  #if EXPECT_COMPILE_ERROR == __LINE__
    d1.clone();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    d2.clone();
  #endif

  #if EXPECT_COMPILE_ERROR == __LINE__
    d5.clone();
  #endif
}

TEST_F(CloneTest, DeviceToDeviceOnDevice) {
}
