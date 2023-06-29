// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>
#include <cuda_runtime.h>


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int r = RUN_ALL_TESTS();
  cudaDeviceReset();  // For cuda-memcheck (https://docs.nvidia.com/cuda/cuda-memcheck/index.html#leak-checking)
  return r;
}
