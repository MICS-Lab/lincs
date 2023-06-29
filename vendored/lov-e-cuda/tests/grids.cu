// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include "../lov-e.hpp"


bool operator==(const Grid& left, const Grid& right) {
  return left.gridDim.x == right.gridDim.x
    && left.gridDim.y == right.gridDim.y
    && left.gridDim.z == right.gridDim.z
    && left.blockDim.x == right.blockDim.x
    && left.blockDim.y == right.blockDim.y
    && left.blockDim.z == right.blockDim.z;
}

std::ostream& operator<<(std::ostream& os, const Grid& grid) {
  os << "{("
    << grid.gridDim.x << ", " << grid.gridDim.y << ", " << grid.gridDim.z
    << "), ("
    << grid.blockDim.x << ", " << grid.blockDim.y << ", " << grid.blockDim.z
    << ")}";

  return os;
}

typedef GridFactory1D<17> grid_1d;
static_assert(grid_1d::blockDim.x == 17);
static_assert(grid_1d::blockDim.y == 1);
static_assert(grid_1d::blockDim.z == 1);

typedef GridFactory2D<11, 13> grid_2d;
static_assert(grid_2d::blockDim.x == 11);
static_assert(grid_2d::blockDim.y == 13);
static_assert(grid_2d::blockDim.z == 1);

typedef GridFactory3D<3, 5, 7> grid_3d;
static_assert(grid_3d::blockDim.x == 3);
static_assert(grid_3d::blockDim.y == 5);
static_assert(grid_3d::blockDim.z == 7);

TEST(GridTest, MakeGrid1D) {
  ASSERT_EQ(grid_1d::make(0), (Grid{{0, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_1d::make(1), (Grid{{1, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_1d::make(17), (Grid{{1, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_1d::make(18), (Grid{{2, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_1d::make(34), (Grid{{2, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_1d::make(35), (Grid{{3, 1, 1}, {17, 1, 1}}));
}

TEST(GridTest, MakeGrid2D) {
  ASSERT_EQ(grid_2d::make(0, 1), (Grid{{0, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 1), (Grid{{1, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(11, 1), (Grid{{1, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(12, 1), (Grid{{2, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(22, 1), (Grid{{2, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(23, 1), (Grid{{3, 1, 1}, {11, 13, 1}}));

  ASSERT_EQ(grid_2d::make(1, 0), (Grid{{1, 0, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 1), (Grid{{1, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 13), (Grid{{1, 1, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 14), (Grid{{1, 2, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 26), (Grid{{1, 2, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_2d::make(1, 27), (Grid{{1, 3, 1}, {11, 13, 1}}));
}

TEST(GridTest, MakeGrid3D) {
  ASSERT_EQ(grid_3d::make(0, 1, 1), (Grid{{0, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 1), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(3, 1, 1), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(4, 1, 1), (Grid{{2, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(6, 1, 1), (Grid{{2, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(7, 1, 1), (Grid{{3, 1, 1}, {3, 5, 7}}));

  ASSERT_EQ(grid_3d::make(1, 0, 1), (Grid{{1, 0, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 1), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 5, 1), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 6, 1), (Grid{{1, 2, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 10, 1), (Grid{{1, 2, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 11, 1), (Grid{{1, 3, 1}, {3, 5, 7}}));

  ASSERT_EQ(grid_3d::make(1, 1, 0), (Grid{{1, 1, 0}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 1), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 7), (Grid{{1, 1, 1}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 8), (Grid{{1, 1, 2}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 14), (Grid{{1, 1, 2}, {3, 5, 7}}));
  ASSERT_EQ(grid_3d::make(1, 1, 15), (Grid{{1, 1, 3}, {3, 5, 7}}));
}

TEST(GridTest, Fixed) {
  ASSERT_EQ(grid_1d::fixed(42), (Grid{{42, 1, 1}, {17, 1, 1}}));
  ASSERT_EQ(grid_2d::fixed(43, 65), (Grid{{43, 65, 1}, {11, 13, 1}}));
  ASSERT_EQ(grid_3d::fixed(47, 78, 91), (Grid{{47, 78, 91}, {3, 5, 7}}));
}

__global__ void kernel_1d() {
  assert(grid_1d::x() == blockIdx.x * blockDim.x + threadIdx.x);
}

TEST(GridTest, Launch1D) {
  auto grid = grid_1d::make(42);
  kernel_1d<<<LOVE_CONFIG(grid)>>>();
  check_last_cuda_error_sync_device();
}

__global__ void kernel_2d() {
  assert(grid_2d::x() == blockIdx.x * blockDim.x + threadIdx.x);
  assert(grid_2d::y() == blockIdx.y * blockDim.y + threadIdx.y);
}

TEST(GridTest, Launch2D) {
  auto grid = grid_2d::make(42, 65);
  kernel_2d<<<LOVE_CONFIG(grid)>>>();
  check_last_cuda_error_sync_device();
}

__global__ void kernel_3d() {
  assert(grid_3d::x() == blockIdx.x * blockDim.x + threadIdx.x);
  assert(grid_3d::y() == blockIdx.y * blockDim.y + threadIdx.y);
  assert(grid_3d::z() == blockIdx.z * blockDim.z + threadIdx.z);
}

TEST(GridTest, Launch3D) {
  auto grid = grid_3d::make(42, 65, 53);
  kernel_3d<<<LOVE_CONFIG(grid)>>>();
  check_last_cuda_error_sync_device();
}
