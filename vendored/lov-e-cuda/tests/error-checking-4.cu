// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsTest, AssertInKernelDetectedOnHost) {
  ASSERT_NO_THROW(check_last_cuda_error_sync_device());

  kernel_assert_false<<<1, 1>>>();
  cudaDeviceSynchronize();
  EXPECT_THROW({
    try {
# 42 "bar/baz.cu" 1
      check_cuda_error(cudaGetLastError());
    } catch(const CudaError& ex) {
      EXPECT_STREQ(
        ex.what(),
        "CUDA ERROR, detected at bar/baz.cu:49: code 710=cudaErrorAssert: device-side assert triggered");
      throw;
    }
  }, CudaError);
}
