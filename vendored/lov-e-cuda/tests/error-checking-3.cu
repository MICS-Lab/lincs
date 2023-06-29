// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


__global__ void kernel_assert_false() {
  assert(false);
}

TEST(CheckCudaErrorsNoSyncTest, AssertInKernelDetectedOnHost) {
  ASSERT_NO_THROW(check_last_cuda_error_sync_device());

  kernel_assert_false<<<1, 1>>>();
  // At this point, `check_last_cuda_error_no_sync` might or might not throw,
  // depending on how quickly the device (or stream) synchronizes (race condition).
  // So we need to synchronize it:
  cudaStreamSynchronize(cudaStreamDefault);
  // ... before calling `check_last_cuda_error_no_sync`
  EXPECT_THROW(check_last_cuda_error_no_sync(), CudaError);
}
