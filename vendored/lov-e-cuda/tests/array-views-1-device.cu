// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


__global__ void kernel_DeviceArrayView1DTest_SetUp(unsigned s0, int* memory) {
  for (unsigned i0 = 0; i0 != s0; ++i0) {
    memory[i0] = 3 * i0;
  }
}

class DeviceArrayView1DTest : public testing::Test {
 protected:
  DeviceArrayView1DTest() : memory(Device::alloc<int>(s0)), array(s0, memory), const_array(s0, memory) {
    kernel_DeviceArrayView1DTest_SetUp<<<1, 1>>>(s0, memory);
  }

  ~DeviceArrayView1DTest() {
    Device::free(memory);
  }

  static const unsigned s0 = 5;
  int* memory;
  ArrayView1D<Device, int> array;
  ArrayView1D<Device, const int> const_array;
};

const unsigned DeviceArrayView1DTest::s0;

TEST_F(DeviceArrayView1DTest, GetSizes) {
  EXPECT_EQ(array.s0(), s0);
  EXPECT_EQ(array.total_size(), s0);
}

TEST_F(DeviceArrayView1DTest, Data) {
  EXPECT_EQ(array.data(), memory);
}

__global__ void kernel_DeviceArrayView1DTest_Index(ArrayView1D<Device, int> array) {
  // array[i0] == memory[i0]
  assert(array[0] == 0);
  assert(array[1] == 3);
  assert(array[2] == 6);
  assert(array[3] == 9);
  assert(array[4] == 12);
}

TEST_F(DeviceArrayView1DTest, Index) {
  kernel_DeviceArrayView1DTest_Index<<<1, 1>>>(array);
  check_last_cuda_error_sync_device();
}

__global__ void kernel_DeviceArrayView1DTest_ConvertToConst(ArrayView1D<Device, int> array) {
  // Can convert to const
  ArrayView1D<Device, const int> const_array(array);

  // Can read from const
  assert(const_array[2] == 6);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    const_array[2] = 65;
  #endif

  // Can't convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView1D<Device, int> non_const_array(const_array);
  #endif
}

TEST_F(DeviceArrayView1DTest, ConvertToConst) {
  kernel_DeviceArrayView1DTest_ConvertToConst<<<1, 1>>>(array);
  check_last_cuda_error_sync_device();

  // Can convert to const
  ArrayView1D<Device, const int> const_array(array);

  // Can't convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView1D<Device, int> non_const_array(const_array);
  #endif
}

__global__ void kernel_DeviceArrayView1DTest_Assign(ArrayView1D<Device, int> array) {
  ArrayView1D<Device, int> other_array(0, nullptr);

  // Can be assigned (with "non-owning pointer" semantics)
  other_array = array;

  assert(other_array.s0() == array.s0());
  assert(other_array.data() == array.data());
  assert(other_array[3] == 9);

  // Can't be assigned if dimensions don't match
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView2D<Device, int>(0, 0, nullptr);
  #endif
}

TEST_F(DeviceArrayView1DTest, Assign) {
  kernel_DeviceArrayView1DTest_Assign<<<1, 1>>>(array);
  check_last_cuda_error_sync_device();

  ArrayView1D<Device, int> other_array(0, nullptr);

  // Can be assigned (with "non-owning pointer" semantics)
  other_array = array;
  EXPECT_EQ(other_array.s0(), s0);
  EXPECT_EQ(other_array.data(), memory);

  // Can't be assigned if dimensions don't match
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView2D<Device, int>(0, 0, nullptr);
  #endif
}

__global__ void kernel_DeviceArrayView1DTest_AssignToConst(ArrayView1D<Device, int> array) {
  ArrayView1D<Device, const int> const_a(0, nullptr);

  // Can be assigned
  const_a = array;
  assert(const_a[3] == 9);

  // Can't be re-assigned to non-const
  ArrayView1D<Device, int> non_const_a(0, nullptr);
  #if EXPECT_COMPILE_ERROR == __LINE__
    non_const_a = const_a;
  #endif
}

TEST_F(DeviceArrayView1DTest, AssignToConst) {
  kernel_DeviceArrayView1DTest_AssignToConst<<<1, 1>>>(array);
  check_last_cuda_error_sync_device();

  ArrayView1D<Device, const int> const_a(0, nullptr);

  // Can be assigned
  const_a = array;

  // Can't be re-assigned to non-const
  ArrayView1D<Device, int> non_const_a(0, nullptr);
  #if EXPECT_COMPILE_ERROR == __LINE__
    non_const_a = const_a;
  #endif
}

__global__ void kernel_DeviceArrayView1DTest_Copy_1(ArrayView1D<Device, int> other_array) {
  other_array[0] = 42;
  other_array[4] = 42;
}

__global__ void kernel_DeviceArrayView1DTest_Copy_2(ArrayView1D<Device, const int> other_array) {
  assert(other_array[0] == 0);
  assert(other_array[4] == 12);
}

TEST_F(DeviceArrayView1DTest, Copy) {
  int* other_memory = Device::alloc<int>(s0);
  ArrayView1D<Device, int> other_array(s0, other_memory);
  kernel_DeviceArrayView1DTest_Copy_1<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();

  copy(array, other_array);

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();

  Device::free(other_memory);
}

TEST_F(DeviceArrayView1DTest, CloneTo) {
  Array1D<Device, int> other_array = array.clone_to<Device>();

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();
}

TEST_F(DeviceArrayView1DTest, Clone) {
  Array1D<Device, int> other_array = array.clone();

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();
}

TEST_F(DeviceArrayView1DTest, CopyConst) {
  int* other_memory = Device::alloc<int>(s0);
  ArrayView1D<Device, int> other_array(s0, other_memory);
  kernel_DeviceArrayView1DTest_Copy_1<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();

  copy(const_array, other_array);

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();

  Device::free(other_memory);
}

TEST_F(DeviceArrayView1DTest, CloneToConst) {
  Array1D<Device, int> other_array = const_array.clone_to<Device>();

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();
}

TEST_F(DeviceArrayView1DTest, CloneConst) {
  Array1D<Device, int> other_array = const_array.clone();

  kernel_DeviceArrayView1DTest_Copy_2<<<1, 1>>>(other_array);
  check_last_cuda_error_sync_device();
}
