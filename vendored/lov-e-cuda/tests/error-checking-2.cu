// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsTest, AssertInKernelDetectedOnHostInFileWithLongName) {
  ASSERT_NO_THROW(check_last_cuda_error_sync_device());

  kernel_assert_false<<<1, 1>>>();
  EXPECT_THROW({
    try {
# 42 "a23456789/b23456789/c23456789/d23456789/e23456789/f23456789/g23456789/h23456789/i23456789/j23456789/k23456789/l23456789/m23456789/n23456789/o23456789/p23456789/q23456789/r23456789/s23456789/bar.cu" 1  // NOLINT(whitespace/line_length)
      check_last_cuda_error_sync_device();
    } catch(const CudaError& ex) {
      EXPECT_STREQ(
        ex.what(),
        // Error is truncated because of long file name
        "CUDA ERROR, detected at a23456789/b23456789/c23456789/d23456789/e23456789/f23456789/"
        "g23456789/h23456789/i23456789/j23456789/k23456789/l23456789/m23456789/n23456789/"
        "o23456789/p23456789/q23456789/r23456789/s23456789/"
        "bar.cu:53: code 710=cudaErrorAssert: devi");
      throw;
    }
  }, CudaError);
}
