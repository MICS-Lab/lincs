// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

#include <gtest/gtest.h>

#include <cassert>

#include "../lov-e.hpp"


class ArrayView2DTest : public testing::Test {
 protected:
  ArrayView2DTest() : array(s1, s0, memory), const_array(s1, s0, memory) {
    for (unsigned i = 0; i != s1 * s0; ++i) {
      memory[i] = 3 * i;
    }
  }

  static const unsigned s1 = 4;
  static const unsigned s0 = 3;
  int memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> array;
  ArrayView2D<Host, const int> const_array;
};

const unsigned ArrayView2DTest::s1;
const unsigned ArrayView2DTest::s0;

TEST_F(ArrayView2DTest, GetSizes) {
  EXPECT_EQ(array.s0(), s0);
  EXPECT_EQ(array.s1(), s1);
  EXPECT_EQ(array.total_size(), s1 * s0);
}

TEST_F(ArrayView2DTest, Data) {
  EXPECT_EQ(array.data(), memory);
}

TEST_F(ArrayView2DTest, Index) {
  // array[i1][i0] == memory[i1 * s0 + i0]
  EXPECT_EQ(array[0][0], 0);
  EXPECT_EQ(array[0][1], 3);
  EXPECT_EQ(array[0][2], 6);
  EXPECT_EQ(array[1][0], 9);
  EXPECT_EQ(array[1][1], 12);
  EXPECT_EQ(array[1][2], 15);
  EXPECT_EQ(array[2][0], 18);
  EXPECT_EQ(array[2][1], 21);
  EXPECT_EQ(array[2][2], 24);
  EXPECT_EQ(array[3][0], 27);
  EXPECT_EQ(array[3][1], 30);
  EXPECT_EQ(array[3][2], 33);
}

TEST_F(ArrayView2DTest, ConvertToConst) {
  // Can convert to const
  ArrayView2D<Host, const int> const_array(array);

  // Can read from const
  EXPECT_EQ(const_array[2][1], 21);

  // Can't write to a const
  #if EXPECT_COMPILE_ERROR == __LINE__
    const_array[2][1] = 65;
  #endif

  // Can't convert back to non-const
  #if EXPECT_COMPILE_ERROR == __LINE__
    ArrayView2D<Host, int> non_const_array(const_array);
  #endif
}

TEST_F(ArrayView2DTest, IndexOnce) {
  ArrayView1D<Host, const int> array_1 = array[1];
  EXPECT_EQ(array_1[0], 9);
  EXPECT_EQ(array_1[1], 12);
  EXPECT_EQ(array_1[2], 15);
}

TEST_F(ArrayView2DTest, Assign) {
  ArrayView2D<Host, int> other_array(0, 0, nullptr);

  // Can be assigned (with "non-owning pointer" semantics)
  other_array = array;
  EXPECT_EQ(other_array.s1(), s1);
  EXPECT_EQ(other_array.s0(), s0);
  EXPECT_EQ(other_array.data(), memory);
  EXPECT_EQ(other_array[3][1], 30);

  // Can't be assigned if dimensions don't match
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView1D<Host, int>(0, nullptr);
  #endif
  #if EXPECT_COMPILE_ERROR == __LINE__
    other_array = ArrayView3D<Host, int>(0, 0, 0, nullptr);
  #endif
}

TEST_F(ArrayView2DTest, AssignToConst) {
  ArrayView2D<Host, const int> const_a(0, 0, nullptr);

  // Can be assigned
  const_a = array;
  EXPECT_EQ(const_a[3][1], 30);

  // Can't be re-assigned to non-const
  ArrayView2D<Host, int> non_const_a(0, 0, nullptr);
  #if EXPECT_COMPILE_ERROR == __LINE__
    non_const_a = const_a;
  #endif
}

TEST_F(ArrayView2DTest, CopyToArrayView) {
  int other_memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> other_array(s1, s0, other_memory);
  other_array[0][0] = 42;
  other_array[3][2] = 42;
  ArrayView2D<Host, const int> const_other_array(s1, s0, other_memory);

  copy(array, other_array);
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(array, const_other_array);
  #endif

  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CopyToRefOfArrayView) {
  int other_memory[s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView2D<Host, int> other_array(s1, s0, other_memory);
  other_array[0][0] = 42;
  other_array[3][2] = 42;
  ArrayView2D<Host, const int> const_other_array(s1, s0, other_memory);

  copy(array, ref(other_array));
  #if EXPECT_COMPILE_ERROR == __LINE__
    copy(array, ref(const_other_array));
  #endif

  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CopyToArray) {
  Array2D<Host, int> other_array(s1, s0, uninitialized);
  other_array[0][0] = 42;
  other_array[3][2] = 42;

  copy(array, ref(other_array));

  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CloneTo) {
  Array2D<Host, int> other_array = array.clone_to<Host>();
  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, Clone) {
  Array2D<Host, int> other_array = array.clone();
  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CopyConst) {
  Array2D<Host, int> other_array(s1, s0, uninitialized);
  other_array[0][0] = 42;
  other_array[3][2] = 42;

  copy(const_array, ref(other_array));

  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CloneToConst) {
  Array2D<Host, int> other_array = const_array.clone_to<Host>();
  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

TEST_F(ArrayView2DTest, CloneConst) {
  Array2D<Host, int> other_array = const_array.clone();
  EXPECT_EQ(other_array[0][0], 0);
  EXPECT_EQ(other_array[3][2], 33);
}

class ArrayView4DTest : public testing::Test {
 protected:
  ArrayView4DTest() : array(s3, s2, s1, s0, memory) {
    for (unsigned i = 0; i != s3 * s2 * s1 * s0; ++i) {
      memory[i] = i;
    }
  }

  static const unsigned s3 = 5;
  static const unsigned s2 = 4;
  static const unsigned s1 = 3;
  static const unsigned s0 = 2;
  int memory[s3 * s2 * s1 * s0];  // NOLINT(runtime/arrays)
  ArrayView4D<Host, int> array;
};

const unsigned ArrayView4DTest::s3;
const unsigned ArrayView4DTest::s2;
const unsigned ArrayView4DTest::s1;
const unsigned ArrayView4DTest::s0;

TEST_F(ArrayView4DTest, Index) {
  EXPECT_EQ(array[0][0][0][0], 0);
  EXPECT_EQ(array[0][0][0][1], 1);
  EXPECT_EQ(array[0][0][1][0], 2);
  EXPECT_EQ(array[0][0][1][1], 3);
  EXPECT_EQ(array[0][0][2][0], 4);
  EXPECT_EQ(array[0][0][2][1], 5);
  EXPECT_EQ(array[0][1][0][0], 6);
  EXPECT_EQ(array[0][1][0][1], 7);
  EXPECT_EQ(array[0][1][1][0], 8);
  EXPECT_EQ(array[0][1][1][1], 9);
  EXPECT_EQ(array[0][1][2][0], 10);
  EXPECT_EQ(array[0][1][2][1], 11);
  EXPECT_EQ(array[0][2][0][0], 12);
  EXPECT_EQ(array[0][2][0][1], 13);
  EXPECT_EQ(array[0][2][1][0], 14);
  EXPECT_EQ(array[0][2][1][1], 15);
  EXPECT_EQ(array[0][2][2][0], 16);
  EXPECT_EQ(array[0][2][2][1], 17);
  EXPECT_EQ(array[0][3][0][0], 18);
  EXPECT_EQ(array[0][3][0][1], 19);
  EXPECT_EQ(array[0][3][1][0], 20);
  EXPECT_EQ(array[0][3][1][1], 21);
  EXPECT_EQ(array[0][3][2][0], 22);
  EXPECT_EQ(array[0][3][2][1], 23);
  EXPECT_EQ(array[1][0][0][0], 24);
  EXPECT_EQ(array[1][0][0][1], 25);
  EXPECT_EQ(array[1][0][1][0], 26);
  EXPECT_EQ(array[1][0][1][1], 27);
  EXPECT_EQ(array[1][0][2][0], 28);
  EXPECT_EQ(array[1][0][2][1], 29);
  EXPECT_EQ(array[1][1][0][0], 30);
  EXPECT_EQ(array[1][1][0][1], 31);
  EXPECT_EQ(array[1][1][1][0], 32);
  EXPECT_EQ(array[1][1][1][1], 33);
  EXPECT_EQ(array[1][1][2][0], 34);
  EXPECT_EQ(array[1][1][2][1], 35);
  EXPECT_EQ(array[1][2][0][0], 36);
  EXPECT_EQ(array[1][2][0][1], 37);
  EXPECT_EQ(array[1][2][1][0], 38);
  EXPECT_EQ(array[1][2][1][1], 39);
  EXPECT_EQ(array[1][2][2][0], 40);
  EXPECT_EQ(array[1][2][2][1], 41);
  EXPECT_EQ(array[1][3][0][0], 42);
  EXPECT_EQ(array[1][3][0][1], 43);
  EXPECT_EQ(array[1][3][1][0], 44);
  EXPECT_EQ(array[1][3][1][1], 45);
  EXPECT_EQ(array[1][3][2][0], 46);
  EXPECT_EQ(array[1][3][2][1], 47);
  EXPECT_EQ(array[2][0][0][0], 48);
  EXPECT_EQ(array[2][0][0][1], 49);
  EXPECT_EQ(array[2][0][1][0], 50);
  EXPECT_EQ(array[2][0][1][1], 51);
  EXPECT_EQ(array[2][0][2][0], 52);
  EXPECT_EQ(array[2][0][2][1], 53);
  EXPECT_EQ(array[2][1][0][0], 54);
  EXPECT_EQ(array[2][1][0][1], 55);
  EXPECT_EQ(array[2][1][1][0], 56);
  EXPECT_EQ(array[2][1][1][1], 57);
  EXPECT_EQ(array[2][1][2][0], 58);
  EXPECT_EQ(array[2][1][2][1], 59);
  EXPECT_EQ(array[2][2][0][0], 60);
  EXPECT_EQ(array[2][2][0][1], 61);
  EXPECT_EQ(array[2][2][1][0], 62);
  EXPECT_EQ(array[2][2][1][1], 63);
  EXPECT_EQ(array[2][2][2][0], 64);
  EXPECT_EQ(array[2][2][2][1], 65);
  EXPECT_EQ(array[2][3][0][0], 66);
  EXPECT_EQ(array[2][3][0][1], 67);
  EXPECT_EQ(array[2][3][1][0], 68);
  EXPECT_EQ(array[2][3][1][1], 69);
  EXPECT_EQ(array[2][3][2][0], 70);
  EXPECT_EQ(array[2][3][2][1], 71);
  EXPECT_EQ(array[3][0][0][0], 72);
  EXPECT_EQ(array[3][0][0][1], 73);
  EXPECT_EQ(array[3][0][1][0], 74);
  EXPECT_EQ(array[3][0][1][1], 75);
  EXPECT_EQ(array[3][0][2][0], 76);
  EXPECT_EQ(array[3][0][2][1], 77);
  EXPECT_EQ(array[3][1][0][0], 78);
  EXPECT_EQ(array[3][1][0][1], 79);
  EXPECT_EQ(array[3][1][1][0], 80);
  EXPECT_EQ(array[3][1][1][1], 81);
  EXPECT_EQ(array[3][1][2][0], 82);
  EXPECT_EQ(array[3][1][2][1], 83);
  EXPECT_EQ(array[3][2][0][0], 84);
  EXPECT_EQ(array[3][2][0][1], 85);
  EXPECT_EQ(array[3][2][1][0], 86);
  EXPECT_EQ(array[3][2][1][1], 87);
  EXPECT_EQ(array[3][2][2][0], 88);
  EXPECT_EQ(array[3][2][2][1], 89);
  EXPECT_EQ(array[3][3][0][0], 90);
  EXPECT_EQ(array[3][3][0][1], 91);
  EXPECT_EQ(array[3][3][1][0], 92);
  EXPECT_EQ(array[3][3][1][1], 93);
  EXPECT_EQ(array[3][3][2][0], 94);
  EXPECT_EQ(array[3][3][2][1], 95);
  EXPECT_EQ(array[4][0][0][0], 96);
  EXPECT_EQ(array[4][0][0][1], 97);
  EXPECT_EQ(array[4][0][1][0], 98);
  EXPECT_EQ(array[4][0][1][1], 99);
  EXPECT_EQ(array[4][0][2][0], 100);
  EXPECT_EQ(array[4][0][2][1], 101);
  EXPECT_EQ(array[4][1][0][0], 102);
  EXPECT_EQ(array[4][1][0][1], 103);
  EXPECT_EQ(array[4][1][1][0], 104);
  EXPECT_EQ(array[4][1][1][1], 105);
  EXPECT_EQ(array[4][1][2][0], 106);
  EXPECT_EQ(array[4][1][2][1], 107);
  EXPECT_EQ(array[4][2][0][0], 108);
  EXPECT_EQ(array[4][2][0][1], 109);
  EXPECT_EQ(array[4][2][1][0], 110);
  EXPECT_EQ(array[4][2][1][1], 111);
  EXPECT_EQ(array[4][2][2][0], 112);
  EXPECT_EQ(array[4][2][2][1], 113);
  EXPECT_EQ(array[4][3][0][0], 114);
  EXPECT_EQ(array[4][3][0][1], 115);
  EXPECT_EQ(array[4][3][1][0], 116);
  EXPECT_EQ(array[4][3][1][1], 117);
  EXPECT_EQ(array[4][3][2][0], 118);
  EXPECT_EQ(array[4][3][2][1], 119);
}
