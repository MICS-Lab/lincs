// Copyright 2021-2022 Vincent Jacques

#include <gtest/gtest.h>

#include "matrix-view.hpp"


int data[125];


TEST(MatrixView1D, CreateAndAccess) {
  MatrixView1D<int> m1(5, data);
  // Writable
  EXPECT_EQ(m1[4] = 42, 42);
  // Readable
  EXPECT_EQ(m1.s0(), 5);
  EXPECT_EQ(m1[4], 42);
}

TEST(MatrixView1D, AccessAsConst) {
  MatrixView1D<int> m1(5, data);
  m1[4] = 42;

  // Convertible to const
  MatrixView1D<const int> m2(m1);
  // Readable
  EXPECT_EQ(m2.s0(), 5);
  EXPECT_EQ(m2[4], 42);

  // Not writable
  #if EXPECT_COMPILE_ERROR == __LINE__
  m2[3] = 57;
  #endif

  // Not convertible back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
  MatrixView1D<int> m3(m2);
  #endif
}

TEST(MatrixView1D, Copy) {
  MatrixView1D<int> m1(5, data);
  m1[4] = 42;

  // Copyable
  MatrixView1D<int> m2(m1);
  // Readable
  EXPECT_EQ(m2.s0(), 5);
  EXPECT_EQ(m2[4], 42);
  // Writable
  EXPECT_EQ(m2[3] = 57, 57);
  // Writes are visible throught the original
  EXPECT_EQ(m1[3], 57);
}

TEST(MatrixView3D, CreateAndAccess) {
  MatrixView3D<int> m1(5, 4, 3, data);
  // Writable
  EXPECT_EQ(m1[4][3][2] = 42, 42);
  // Readable
  EXPECT_EQ(m1.s2(), 5);
  EXPECT_EQ(m1.s1(), 4);
  EXPECT_EQ(m1.s0(), 3);
  EXPECT_EQ(m1[4][3][2], 42);
}

TEST(MatrixView3D, AccessAsConst) {
  MatrixView3D<int> m1(5, 4, 3, data);
  m1[4][3][2] = 42;

  // Convertible to const
  MatrixView3D<const int> m2(m1);
  // Readable
  EXPECT_EQ(m2.s2(), 5);
  EXPECT_EQ(m2.s1(), 4);
  EXPECT_EQ(m2.s0(), 3);
  EXPECT_EQ(m2[4][3][2], 42);
  // Not writable
  #if EXPECT_COMPILE_ERROR == __LINE__
  m2[3][2][1] = 57;
  #endif

  // Not convertible back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
  MatrixView3D<int> m3(m2);
  #endif
}

TEST(MatrixView3D, Copy) {
  MatrixView3D<int> m1(5, 4, 3, data);
  m1[4][3][2] = 42;

  // Copyable
  MatrixView3D<int> m2(m1);
  // Readable
  EXPECT_EQ(m2.s2(), 5);
  EXPECT_EQ(m2.s1(), 4);
  EXPECT_EQ(m2.s0(), 3);
  EXPECT_EQ(m2[4][3][2], 42);
  // Writable
  EXPECT_EQ(m2[3][2][1] = 57, 57);
  // Writes are visible throught the original
  EXPECT_EQ(m1[3][2][1], 57);
}
