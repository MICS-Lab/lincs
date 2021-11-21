// Copyright 2021 Vincent Jacques

#ifndef MATRIX_VIEW_HPP_
#define MATRIX_VIEW_HPP_

#include <cassert>


/*
A set of small wrappers that have the same ownership semantics as a pointer,
and provide a multi-dimensional array interface.

Usable in __host__, __device__, and __global__ functions.

Ownership semantics:
  - a copy of a MatrixView points at the same data (like pointers)
  - a MatrixView1D<const T> forbids modification of underlying data (like a const T*)
  - MatrixView1D<T> is implicitely convertible to MatrixView1D<const T> (like pointers)
  - MatrixView1D<T> is implicitely convertible to const MatrixView1D<T> (like everything)
  - data must be managed (allocated, released) outside this class
  - the size of usable data must be at least sN * ... * s0 * sizeof(T)

Multi-dimensional array interface:
  MatrixView3D<int> a(4, 5, 6);
  a[3][4][5] = 42;
*/

#define DECS __host__ __device__

template<typename T>
class MatrixView1D {
 public:
  DECS MatrixView1D(int s0, T* data) : _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView1D;

  template<typename U>
  DECS MatrixView1D(const MatrixView1D<U>& o) : _s0(o._s0), _data(o._data) {}

  DECS int s0() const { return _s0; }

  DECS T& operator[](int i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

 private:
  const int _s0;
  T* const _data;
};


template<typename T>
class MatrixView2D {
 public:
  DECS MatrixView2D(int s1, int s0, T* data) : _s1(s1), _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView2D;

  template<typename U>
  DECS MatrixView2D(const MatrixView2D<U>& o) : _s1(o._s1), _s0(o._s0), _data(o._data) {}

  DECS int s1() const { return _s1; }
  DECS int s0() const { return _s0; }

  DECS MatrixView1D<T> operator[](int i1) const {
    assert(i1 < _s1);
    return MatrixView1D<T>(_s0, _data + i1 * _s0);
  }

 private:
  const int _s1;
  const int _s0;
  T* const _data;
};


template<typename T>
class MatrixView3D {
 public:
  DECS MatrixView3D(int s2, int s1, int s0, T* data) : _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView3D;

  template<typename U>
  DECS MatrixView3D(const MatrixView3D<U>& o) : _s2(o._s2), _s1(o._s1), _s0(o._s0), _data(o._data) {}

  DECS int s2() const { return _s2; }
  DECS int s1() const { return _s1; }
  DECS int s0() const { return _s0; }

  DECS MatrixView2D<T> operator[](int i2) const {
    assert(i2 < _s2);
    return MatrixView2D<T>(_s1, _s0, _data + i2 * _s1 * _s0);
  }

 private:
  const int _s2;
  const int _s1;
  const int _s0;
  T* const _data;
};

#undef DECS

#endif  // MATRIX_VIEW_HPP_
