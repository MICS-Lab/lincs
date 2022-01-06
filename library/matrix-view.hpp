// Copyright 2021-2022 Vincent Jacques

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
  int* data = ...;
  MatrixView3D a(4, 5, 6, data);
  a[3][4][5] = 42;

*/

#ifndef MATRIX_VIEW_HPP_
#define MATRIX_VIEW_HPP_

#include <cassert>

#include "cuda-utils.hpp"


template<typename T>
class MatrixView1D {
 public:
  HOST_DEVICE_DECORATORS
  MatrixView1D(unsigned int s0, T* data) : _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView1D;

  template<typename U>
  HOST_DEVICE_DECORATORS
  MatrixView1D(const MatrixView1D<U>& o) : _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned int s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  T& operator[](unsigned int i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

 private:
  const unsigned int _s0;
  T* const _data;
};


template<typename T>
class MatrixView2D {
 public:
  HOST_DEVICE_DECORATORS
  MatrixView2D(unsigned int s1, unsigned int s0, T* data) : _s1(s1), _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView2D;

  template<typename U>
  HOST_DEVICE_DECORATORS
  MatrixView2D(const MatrixView2D<U>& o) : _s1(o._s1), _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned int s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  unsigned int s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  MatrixView1D<T> operator[](unsigned int i1) const {
    assert(i1 < _s1);
    return MatrixView1D<T>(_s0, _data + i1 * _s0);
  }

 private:
  const unsigned int _s1;
  const unsigned int _s0;
  T* const _data;
};


template<typename T>
class MatrixView3D {
 public:
  HOST_DEVICE_DECORATORS
  MatrixView3D(unsigned int s2, unsigned int s1, unsigned int s0, T* data) :
    _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  template<typename U>
  friend class MatrixView3D;

  template<typename U>
  HOST_DEVICE_DECORATORS
  MatrixView3D(const MatrixView3D<U>& o) : _s2(o._s2), _s1(o._s1), _s0(o._s0), _data(o._data) {}

  HOST_DEVICE_DECORATORS
  unsigned int s2() const { return _s2; }
  HOST_DEVICE_DECORATORS
  unsigned int s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  unsigned int s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  MatrixView2D<T> operator[](unsigned int i2) const {
    assert(i2 < _s2);
    return MatrixView2D<T>(_s1, _s0, _data + i2 * _s1 * _s0);
  }

 private:
  const unsigned int _s2;
  const unsigned int _s1;
  const unsigned int _s0;
  T* const _data;
};

#undef DECS

#endif  // MATRIX_VIEW_HPP_
