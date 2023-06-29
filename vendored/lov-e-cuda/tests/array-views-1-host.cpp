// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


class HostArrayView1DTest : public testing::Test {
 protected:
  HostArrayView1DTest() : array(s0, memory), const_array(s0, memory) {
    for (unsigned i = 0; i != s0; ++i) {
      memory[i] = 3 * i;
    }
  }

  static const unsigned s0 = 5;
  int memory[s0];  // NOLINT(runtime/arrays)
  ArrayView1D<Host, int> array;
  ArrayView1D<Host, const int> const_array;
};

const unsigned HostArrayView1DTest::s0;

TEST_F(HostArrayView1DTest, GetSizes) {
  EXPECT_EQ(array.s0(), s0);
  EXPECT_EQ(array.total_size(), s0);
}

TEST_F(HostArrayView1DTest, Data) {
  EXPECT_EQ(array.data(), memory);
}

TEST_F(HostArrayView1DTest, Index) {
  // array[i0] == memory[i0]
  EXPECT_EQ(array[0], 0);
  EXPECT_EQ(array[1], 3);
  EXPECT_EQ(array[2], 6);
  EXPECT_EQ(array[3], 9);
  EXPECT_EQ(array[4], 12);
}

TEST_F(HostArrayView1DTest, ConvertToConst) {
  // Can convert to const
  ArrayView1D<Host, const int> const_array(array);

  // Can read from const
  EXPECT_EQ(const_array[2], 6);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    const_array[2] = 65;
  #endif

  // Can't convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView1D<Host, int> non_const_array(const_array);
  #endif
}

TEST_F(HostArrayView1DTest, Assign) {
  ArrayView1D<Host, int> other_array(0, nullptr);

  // Can be assigned (with "non-owning pointer" semantics)
  other_array = array;
  EXPECT_EQ(other_array.s0(), s0);
  EXPECT_EQ(other_array.data(), memory);
  EXPECT_EQ(other_array[3], 9);

  // Can't be assigned if dimensions don't match
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView2D<Host, int>(0, 0, nullptr);
  #endif
}

TEST_F(HostArrayView1DTest, AssignToConst) {
  ArrayView1D<Host, const int> const_a(0, nullptr);

  // Can be assigned
  const_a = array;
  EXPECT_EQ(const_a[3], 9);

  // Can't be re-assigned to non-const
  ArrayView1D<Host, int> non_const_a(0, nullptr);
  #if EXPECT_COMPILE_ERROR == __LINE__
    non_const_a = const_a;
  #endif
}

TEST_F(HostArrayView1DTest, Copy) {
  int other_memory[s0];  // NOLINT(runtime/arrays)
  ArrayView1D<Host, int> other_array(s0, other_memory);
  other_array[0] = 42;
  other_array[4] = 42;

  copy(array, other_array);

  EXPECT_EQ(other_array[0], 0);
  EXPECT_EQ(other_array[4], 12);
}

TEST_F(HostArrayView1DTest, CloneTo) {
  Array1D<Host, int> other_array = array.clone_to<Host>();
  EXPECT_EQ(other_array[4], 12);
}

TEST_F(HostArrayView1DTest, Clone) {
  Array1D<Host, int> other_array = array.clone();
  EXPECT_EQ(other_array[4], 12);
}

TEST_F(HostArrayView1DTest, CopyConst) {
  int other_memory[s0];  // NOLINT(runtime/arrays)
  ArrayView1D<Host, int> other_array(s0, other_memory);
  other_array[0] = 42;
  other_array[4] = 42;

  copy(const_array, other_array);

  EXPECT_EQ(other_array[0], 0);
  EXPECT_EQ(other_array[4], 12);
}

TEST_F(HostArrayView1DTest, CloneToConst) {
  Array1D<Host, int> other_array = const_array.clone_to<Host>();
  EXPECT_EQ(other_array[4], 12);
}

TEST_F(HostArrayView1DTest, CloneConst) {
  Array1D<Host, int> other_array = const_array.clone();
  EXPECT_EQ(other_array[4], 12);
}
