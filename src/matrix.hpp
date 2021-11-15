// Copyright 2021 Vincent Jacques

#ifndef MATRIX_HPP_
#define MATRIX_HPP_

#include <cassert>
#include <utility>
#include <type_traits>

#include "cuda-utils.hpp"


struct Host/*Space*/ {
  template<typename T>
  static T* alloc(int n) { return alloc_host<T>(n); }

  template<typename T>
  static T* clone_from_device(int n, T* d) {
    return clone_device_to_host(n, d);
  }

  template<typename T>
  static void free(T* p) { free_host(p); }
};

struct Device/*Space*/ {
  template<typename T>
  static T* alloc(int n) { return alloc_device<T>(n); }

  template<typename T>
  static T* clone_from_host(int n, T* h) {
    return clone_host_to_device(n, h);
  }

  template<typename T>
  static void free(T* p) { free_device(p); }
};

template<typename T>
struct View1D {
  View1D(int s0, T* data) : _s0(s0), _data(data) {}

  T& operator[](int i0) {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  const T& operator[](int i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

 private:
  int _s0;
  T* _data;
};

template<typename Space, typename T>
struct Matrix1D {
  explicit Matrix1D(int s0) : Matrix1D(s0, Space::template alloc<T>(s0)) {}

  // Non-copyable
  Matrix1D(const Matrix1D&) = delete;
  Matrix1D& operator=(const Matrix1D&) = delete;

  // Movable
  Matrix1D(Matrix1D&& o) : _s0(o._s0), _data(std::exchange(o._data, nullptr)) {}
  Matrix1D& operator=(Matrix1D&& o) {
    _s0 = o._s0;
    _data = std::exchange(o._data, nullptr);
    return *this;
  }

  ~Matrix1D() {
    if (_data) Space::free(_data);
  }

  int s0() const { return _s0; }

  T& operator[](int i0) {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  const T& operator[](int i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  Matrix1D<Device, T> clone_to(Device) const {
    return Matrix1D<Device, T>(_s0, Device::clone_from_host(_s0, _data));
  }

  Matrix1D<Host, T> clone_to(Host) const {
    return Matrix1D<Host, T>(_s0, Host::clone_from_device(_s0, _data));
  }

 private:
  template<typename OtherSpace, typename OtherT> friend class Matrix1D;

  explicit Matrix1D(int s0, T* data) : _s0(s0), _data(data) {}

 private:
  int _s0;
  T* _data;
};

template<typename Space, typename T>
struct Matrix2D {
  Matrix2D(int s1, int s0) : Matrix2D(s1, s0, Space::template alloc<T>(s1 * s0)) {}

  // Non-copyable
  Matrix2D(const Matrix2D&) = delete;
  Matrix2D& operator=(const Matrix2D&) = delete;

  // Movable
  Matrix2D(Matrix2D&& o) : _s1(o._s1), _s0(o._s0), _data(std::exchange(o._data, nullptr)) {}
  Matrix2D& operator=(Matrix2D&& o) {
    _s1 = o._s1;
    _s0 = o._s0;
    _data = std::exchange(o._data, nullptr);
    return *this;
  }

  ~Matrix2D() {
    if (_data) Space::free(_data);
  }

  int s0() const { return _s0; }

  int s1() const { return _s1; }

  View1D<T> operator[](int i1) {
    assert(i1 < _s1);
    return View1D<T>(_s0, _data + i1 * _s0);
  }

  const View1D<T> operator[](int i1) const {
    assert(i1 < _s1);
    return View1D<T>(_s0, _data + i1 * _s0);
  }

  Matrix2D<Device, T> clone_to(Device) const {
    return Matrix2D<Device, T>(_s1, _s0, Device::clone_from_host(_s1 * _s0, _data));
  }

  Matrix2D<Host, T> clone_to(Host) const {
    return Matrix2D<Host, T>(_s1, _s0, Host::clone_from_device(_s1 * _s0, _data));
  }

 private:
  template<typename OtherSpace, typename OtherT> friend class Matrix2D;

  explicit Matrix2D(int s1, int s0, T* data) : _s1(s1), _s0(s0), _data(data) {}

 private:
  int _s1;
  int _s0;
  T* _data;
};

template<typename OtherSpace, typename Space, typename T>
typename std::enable_if_t<!std::is_same_v<Space, OtherSpace>, Matrix1D<OtherSpace, T>>
transfer_to(Matrix1D<Space, T>&& m) {
  // Ensure for consistency that the parameter is unusable after transfer
  // (moved to a local temporary that gets destroyed)
  return Matrix1D<Space, T>(std::move(m)).clone_to(OtherSpace());
}

template<typename Space, typename T>
Matrix1D<Space, T>&& transfer_to(Matrix1D<Space, T>&& m) {
  return std::move(m);
}

template<typename OtherSpace, typename Space, typename T>
typename std::enable_if_t<!std::is_same_v<Space, OtherSpace>, Matrix2D<OtherSpace, T>>
transfer_to(Matrix2D<Space, T>&& m) {
  // Ensure for consistency that the parameter is unusable after transfer
  // (moved to a local temporary that gets destroyed)
  return Matrix2D<Space, T>(std::move(m)).clone_to(OtherSpace());
}

template<typename Space, typename T>
Matrix2D<Space, T>&& transfer_to(Matrix2D<Space, T>&& m) {
  return std::move(m);
}

#endif  // MATRIX_HPP_
