// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


// CUDA has a concept of "sticky" error that can only be fixed by exiting the process.
// (https://stackoverflow.com/questions/56329377/reset-cuda-context-after-exception)
// Google Test does not provide an option to run each test in its own process.
// (https://groups.google.com/g/googletestframework/c/VpDFzJEGXOA)
// So, to make sure an error condition created by a test does not affect a subsequent test,
// we put only one test per file, and we run each file as its own process.
// We also make sure that the error we detect is created by the test itself,
// by checking CUDA errors at the beginning of each test in this suite.

__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsTest, AssertInKernelDetectedOnHost) {
  ASSERT_NO_THROW(check_last_cuda_error_sync_device());

  kernel_assert_false<<<1, 1>>>();
  EXPECT_THROW({
    try {
// Use a line directive (https://gcc.gnu.org/onlinedocs/cpp/Preprocessor-Output.html)
// to stabilize the expected exception message.
# 42 "foo/bar.cu" 1
      check_last_cuda_error_sync_device();
    } catch(const CudaError& ex) {
      EXPECT_STREQ(
        ex.what(),
        "CUDA ERROR, detected at foo/bar.cu:49: code 710=cudaErrorAssert: device-side assert triggered");
      throw;
    }
  }, CudaError);
}
