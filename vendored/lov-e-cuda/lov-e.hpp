// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret

// Documentation and license information for this file are available at https://github.com/jacquev6/lov-e-cuda

#ifndef LOV_E_HPP_
#define LOV_E_HPP_

#ifdef LINCS_HAS_NVCC
#include <cuda_runtime.h>
#endif  // LINCS_HAS_NVCC

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <type_traits>


#ifdef __NVCC__
#define HOST_DEVICE_DECORATORS __host__ __device__
#else
#define HOST_DEVICE_DECORATORS
#endif


/*                          *
 * Error checking utilities *
 *                          */

#ifdef LINCS_HAS_NVCC

struct CudaError : public std::exception {
  CudaError(const char* const file_, const unsigned line_, const cudaError_t error_) :
      file(file_), line(line_), error(error_) {
    std::snprintf(
      _what, sizeof(_what),
      "CUDA ERROR, detected at %s:%i: code %i=%s: %s",
      file, line, static_cast<unsigned>(error), cudaGetErrorName(error), cudaGetErrorString(error));
  }

  const char* what() const noexcept override {
    return _what;
  }

  const char* const file;
  const unsigned line;
  const cudaError_t error;

 private:
  char _what[256];
};

// @todo Test the __device__ versions of the error checking functions

HOST_DEVICE_DECORATORS
inline void check_cuda_error_(const cudaError_t error, const char* file, const unsigned line) {
  if (error != cudaSuccess) {
    #ifdef __CUDA_ARCH__
      printf(
        "CUDA ERROR, detected at %s:%i: code %i=%s: %s\n",
        file, line, static_cast<unsigned int>(error), cudaGetErrorName(error), cudaGetErrorString(error));
    #else
      throw CudaError(file, line, error);
    #endif
  }
}

HOST_DEVICE_DECORATORS
inline void check_last_cuda_error_no_sync_(const char* file, const unsigned line) {
  check_cuda_error_(cudaGetLastError(), file, line);
}

HOST_DEVICE_DECORATORS
inline void check_last_cuda_error_sync_device_(const char* const file, const unsigned line) {
  cudaDeviceSynchronize();
  check_last_cuda_error_no_sync_(file, line);
}

HOST_DEVICE_DECORATORS
inline void check_last_cuda_error_sync_stream_(cudaStream_t stream, const char* const file, const unsigned line) {
  cudaStreamSynchronize(stream);
  check_last_cuda_error_no_sync_(file, line);
}

#define check_cuda_error(e) check_cuda_error_(e, __FILE__, __LINE__)

#define check_last_cuda_error_no_sync() check_last_cuda_error_no_sync_(__FILE__, __LINE__)
#define check_last_cuda_error_sync_stream(stream) check_last_cuda_error_sync_stream_(stream, __FILE__, __LINE__)
#define check_last_cuda_error_sync_device() check_last_cuda_error_sync_device_(__FILE__, __LINE__)

#endif  // LINCS_HAS_NVCC

/*                   *
 * Memory management *
 *                   */


class Anywhere {};

class Host {
 public:
  static const bool can_be_allocated_on_device = false;

 private:
  template<typename T>
  static void force_memset(const std::size_t n, const char v, T* const p) {
    std::memset(reinterpret_cast<void*>(p), v, n * sizeof(T));
  }

  template<typename T>
  static T* force_alloc(const std::size_t n) {
    if (n == 0) {
      return nullptr;
    } else {
      T* const p = reinterpret_cast<T*>(std::malloc(n * sizeof(T)));
      #ifndef NDEBUG
        // Attempt to make use of uninitialized memory more noticeable by actually
        // initializing it to a large weird-looking value made of a repeated byte.
        // The repetition of the byte 0x66 yields the following values:
        // - 8-bits integer: 102
        // - 16-bits integer: 26214
        // - 32-bits integer: 1717986918
        // - 64-bits integer: 7378697629483820646
        // - IEEE 754 float: 2.72008e+23
        // - IEEE 754 double: 1.9035985662552932e+185
        force_memset(n, 0x66, p);
      #endif
      return p;
    }
  }

 public:
  template<typename T>
  static void memset(const std::size_t n, const char v, T* const p) {
    static_assert(std::is_trivial<T>::value, "T must be trivial. Workaround: search for 'trivial' in the doc.");
    force_memset<T>(n, v, p);
  }

  template<typename T>
  static void memreset(const std::size_t n, T* const p) {
    memset(n, 0, p);
  }

  template<typename T>
  static T* alloc(const std::size_t n) {
    static_assert(std::is_trivial<T>::value, "T must be trivial. Workaround: search for 'trivial' in the doc.");
    return force_alloc<T>(n);
  }

  template<typename T>
  static T* alloc_zeroed(const std::size_t n) {
    T* const p = alloc<T>(n);
    memreset(n, p);
    return p;
  }

  template<typename T>
  HOST_DEVICE_DECORATORS  // Required in ~Array?D
  static void free(T* p) {
    #ifndef __CUDA_ARCH__
      if (p == nullptr) {
        return;
      } else {
        std::free(p);
      }
    #endif
  }
};

#ifdef LINCS_HAS_NVCC

class Device {
 public:
  static const bool can_be_allocated_on_device = true;

 private:
  template<typename T>
  static void force_memset(const std::size_t n, const char v, T* const p) {
    check_cuda_error(cudaMemset(p, v, n * sizeof(T)));
  }

  template<typename T>
  HOST_DEVICE_DECORATORS
  static T* force_alloc(const std::size_t n) {
    if (n == 0) {
      return nullptr;
    } else {
      T* p;
      check_cuda_error(cudaMalloc(&p, n * sizeof(T)));
      #if !defined(NDEBUG) && !defined(__CUDA_ARCH__)
        // See comment in Host::force_alloc
        force_memset(n, 0x66, p);
      #endif
      return p;
    }
  }

 public:
  template<typename T>
  static void memset(const std::size_t n, const char v, T* const p) {
    static_assert(std::is_trivial<T>::value, "T must be trivial. Workaround: search for 'trivial' in the doc.");
    force_memset<T>(n, v, p);
  }

  template<typename T>
  static void memreset(const std::size_t n, T* const p) {
    memset(n, 0, p);
  }

  template<typename T>
  HOST_DEVICE_DECORATORS
  static T* alloc(const std::size_t n) {
    static_assert(std::is_trivial<T>::value, "T must be trivial. Workaround: search for 'trivial' in the doc.");
    return force_alloc<T>(n);
  }

  template<typename T>
  static T* alloc_zeroed(const std::size_t n) {
    T* const p = alloc<T>(n);
    memreset(n, p);
    return p;
  }

  template<typename T>
  HOST_DEVICE_DECORATORS
  static void free(T* p) {
    if (p == nullptr) {
      return;
    } else {
      check_cuda_error(cudaFree(p));
    }
  }
};

#endif  // LINCS_HAS_NVCC

template<typename WhereFrom>
struct From {
  template<typename WhereTo>
  struct To {
    template<typename T>
    static void copy(const std::size_t n, const T* const src, typename std::remove_const<T>::type* const dst);

    template<typename T>
    static T* clone(const std::size_t n, const T* const src) {
      T* dst = WhereTo::template alloc<T>(n);
      copy(n, src, dst);
      return dst;
    }
  };
};

template<> template<> template<typename T>
void From<Host>::To<Host>::copy(
    const std::size_t n,
    const T* const src,
    typename std::remove_const<T>::type* const dst) {
  if (n == 0) {
    return;
  } else {
    std::memcpy(dst, src, n * sizeof(T));
  }
}

#ifdef LINCS_HAS_NVCC

template<> template<> template<typename T>
void From<Host>::To<Device>::copy(
    const std::size_t n,
    const T* const src,
    typename std::remove_const<T>::type* const dst) {
  if (n == 0) {
    return;
  } else {
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
  }
}

template<> template<> template<typename T>
void From<Device>::To<Device>::copy(
    const std::size_t n,
    const T* const src,
    typename std::remove_const<T>::type* const dst) {
  if (n == 0) {
    return;
  } else {
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToDevice));
  }
}

template<> template<> template<typename T>
void From<Device>::To<Host>::copy(
    const std::size_t n,
    const T* const src,
    typename std::remove_const<T>::type* const dst) {
  if (n == 0) {
    return;
  } else {
    check_cuda_error(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost));
  }
}

#endif  // LINCS_HAS_NVCC

/*                       *
 * Arrays and ArrayViews *
 *                       */

// The 'ArrayView?D' classes have "non-owning pointer" semantics, so they follow the
// [Rule of Zero](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-zero).

// The 'Array?D' classes have "owning pointer" semantics, so they follow the
// [Rule of Five](http://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rc-five).

// The one-dimensional case is special because its operator[] actually accesses the underlying memory.
// Trying to access device memory from the host (and reciprocally) fails at run time. We provide
// specializations for each possible 'Where', to decorate (with '__host__' and/or '__device__') the
// 'T& operator[](unsigned i0)' to forbid that *during compilation*. This implies some code duplication,
// but it's worth it.

// Constructor parameters for 'Array?D'
enum Zeroed {zeroed};
enum Uninitialized {uninitialized};

// BEGIN GENERATED SECTION: arrays-and-array-views

template<typename Where, typename T> class Array1D;

template<typename Where, typename T> class ArrayView1D;

template<typename T>
class ArrayView1D<Anywhere, T> {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView1D(std::size_t s0, T* data) :
    _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Anywhere, U>& o) :
    _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D& operator=(const ArrayView1D<Anywhere, U>& o) {
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView1D<Anywhere, U>() {
    return ArrayView1D<Anywhere, U>(_s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s0; }

  HOST_DEVICE_DECORATORS
  T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

 private:
  std::size_t _s0;
  T* _data;
};

template<typename T>
class ArrayView1D<Host, T> {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView1D(std::size_t s0, T* data) :
    _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Host, U>& o) :
    _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D& operator=(const ArrayView1D<Host, U>& o) {
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView1D<Anywhere, U>() {
    return ArrayView1D<Anywhere, U>(_s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s0; }

  T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Host::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array1D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array1D<Host, typename std::remove_const<T>::type> clone() const { return clone_to<Host>(); }

 private:
  std::size_t _s0;
  T* _data;

  friend class Array1D<Host, typename std::remove_const<T>::type>;
};

#ifdef LINCS_HAS_NVCC

template<typename T>
class ArrayView1D<Device, T> {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView1D(std::size_t s0, T* data) :
    _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D(const ArrayView1D<Device, U>& o) :
    _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView1D& operator=(const ArrayView1D<Device, U>& o) {
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView1D<Anywhere, U>() {
    return ArrayView1D<Anywhere, U>(_s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s0; }

#ifdef __NVCC__
  __device__
  T& operator[](unsigned i0) const {
    assert(i0 < _s0);
    return *(_data + i0);
  }
#endif

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Device::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array1D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array1D<Device, typename std::remove_const<T>::type> clone() const { return clone_to<Device>(); }

 private:
  std::size_t _s0;
  T* _data;

  friend class Array1D<Device, typename std::remove_const<T>::type>;
};

#endif  // LINCS_HAS_NVCC

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView1D<WhereFrom, T> src, ArrayView1D<WhereTo, typename std::remove_const<T>::type> dst) {
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s0(), src.data(), dst.data());
}

template<typename Where, typename T>
class Array1D : public ArrayView1D<Where, const T> {
 public:
  // RAII
  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>
  Array1D(std::size_t s0, Uninitialized) :
    ArrayView1D<Where, const T>(s0, Where::template alloc<T>(s0))
  {}

  template<
    typename = void,
    typename W = Where,
    typename = typename std::enable_if<W::can_be_allocated_on_device>::type
  >
  HOST_DEVICE_DECORATORS
  Array1D(std::size_t s0, Uninitialized) :
    ArrayView1D<Where, const T>(s0, Where::template alloc<T>(s0))
  {}

  Array1D(std::size_t s0, Zeroed) :
    ArrayView1D<Where, const T>(s0, Where::template alloc_zeroed<T>(s0))
  {}

  HOST_DEVICE_DECORATORS
  ~Array1D() {
    free();
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  T* data() const { return const_cast<T*>(this->_data); }

  // @todo Fix decorators: forbid dereferencing host memory on host and vice versa
  HOST_DEVICE_DECORATORS
  T& operator[](unsigned i0) const {
    assert(i0 < this->_s0);
    return *(data() + i0);
  }

  void fill_with_zeros() const {
    Where::template memreset<T>(this->total_size(), data());
  }

  // Not copyable
  Array1D(const Array1D&) = delete;
  Array1D& operator=(const Array1D&) = delete;

  // But movable
  HOST_DEVICE_DECORATORS
  Array1D(Array1D&& o) : ArrayView1D<Where, const T>(o) {
    o._s0 = 0;
    o._data = nullptr;
  }
  HOST_DEVICE_DECORATORS
  Array1D& operator=(Array1D&& o) {
    free();
    static_cast<ArrayView1D<Where, const T>&>(*this) = o;
    o._s0 = 0;
    o._data = nullptr;
    return *this;
  }

 private:
  HOST_DEVICE_DECORATORS
  void free() {
    Where::free(data());
  }
};

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView1D<Where, T> ref(const Array1D<Where, T>& a) {
  return ArrayView1D<Where, T>(a.s0(), a.data());
}

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView1D<Where, T> ref(const ArrayView1D<Where, T>& a) {
  return a;
}

template<typename T>
template<typename WhereTo>
Array1D<WhereTo, typename std::remove_const<T>::type> ArrayView1D<Host, T>::clone_to() const {
  Array1D<WhereTo, typename std::remove_const<T>::type> dst(this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

#ifdef LINCS_HAS_NVCC

template<typename T>
template<typename WhereTo>
Array1D<WhereTo, typename std::remove_const<T>::type> ArrayView1D<Device, T>::clone_to() const {
  Array1D<WhereTo, typename std::remove_const<T>::type> dst(this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

#endif  // LINCS_HAS_NVCC

template<typename Where, typename T> class Array2D;

template<typename Where, typename T>
class ArrayView2D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView2D(std::size_t s1, std::size_t s0, T* data) :
    _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView2D(const ArrayView2D<Where, U>& o) :
    _s1(o.s1()), _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView2D& operator=(const ArrayView2D<Where, U>& o) {
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView2D<Anywhere, U>() {
    return ArrayView2D<Anywhere, U>(_s1, _s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s1 * _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView1D<Where, T> operator[](unsigned i1) const {
    assert(i1 < _s1);
    return ArrayView1D<Where, T>(_s0, _data + i1 * _s0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Where::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array2D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array2D<Where, typename std::remove_const<T>::type> clone() const { return clone_to<Where>(); }

 private:
  std::size_t _s1;
  std::size_t _s0;
  T* _data;

  friend class Array2D<Where, typename std::remove_const<T>::type>;
};

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView2D<WhereFrom, T> src, ArrayView2D<WhereTo, typename std::remove_const<T>::type> dst) {
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s1() * src.s0(), src.data(), dst.data());
}

template<typename Where, typename T>
class Array2D : public ArrayView2D<Where, const T> {
 public:
  // RAII
  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>
  Array2D(std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView2D<Where, const T>(s1, s0, Where::template alloc<T>(s1 * s0))
  {}

  template<
    typename = void,
    typename W = Where,
    typename = typename std::enable_if<W::can_be_allocated_on_device>::type
  >
  HOST_DEVICE_DECORATORS
  Array2D(std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView2D<Where, const T>(s1, s0, Where::template alloc<T>(s1 * s0))
  {}

  Array2D(std::size_t s1, std::size_t s0, Zeroed) :
    ArrayView2D<Where, const T>(s1, s0, Where::template alloc_zeroed<T>(s1 * s0))
  {}

  HOST_DEVICE_DECORATORS
  ~Array2D() {
    free();
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  T* data() const { return const_cast<T*>(this->_data); }

  HOST_DEVICE_DECORATORS
  ArrayView1D<Where, T> operator[](unsigned i1) const {
    assert(i1 < this->_s1);
    return ArrayView1D<Where, T>(
      this->_s0, data() + i1 * this->_s0);
  }

  void fill_with_zeros() const {
    Where::template memreset<T>(this->total_size(), data());
  }

  // Not copyable
  Array2D(const Array2D&) = delete;
  Array2D& operator=(const Array2D&) = delete;

  // But movable
  HOST_DEVICE_DECORATORS
  Array2D(Array2D&& o) : ArrayView2D<Where, const T>(o) {
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
  }
  HOST_DEVICE_DECORATORS
  Array2D& operator=(Array2D&& o) {
    free();
    static_cast<ArrayView2D<Where, const T>&>(*this) = o;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
    return *this;
  }

 private:
  HOST_DEVICE_DECORATORS
  void free() {
    Where::free(data());
  }
};

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView2D<Where, T> ref(const Array2D<Where, T>& a) {
  return ArrayView2D<Where, T>(a.s1(), a.s0(), a.data());
}

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView2D<Where, T> ref(const ArrayView2D<Where, T>& a) {
  return a;
}

template<typename WhereFrom, typename T>
template<typename WhereTo>
Array2D<WhereTo, typename std::remove_const<T>::type> ArrayView2D<WhereFrom, T>::clone_to() const {
  Array2D<WhereTo, typename std::remove_const<T>::type> dst(
    this->s1(), this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

template<typename Where, typename T> class Array3D;

template<typename Where, typename T>
class ArrayView3D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView3D(std::size_t s2, std::size_t s1, std::size_t s0, T* data) :
    _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView3D(const ArrayView3D<Where, U>& o) :
    _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView3D& operator=(const ArrayView3D<Where, U>& o) {
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView3D<Anywhere, U>() {
    return ArrayView3D<Anywhere, U>(_s2, _s1, _s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s2() const { return _s2; }
  HOST_DEVICE_DECORATORS
  std::size_t s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s2 * _s1 * _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView2D<Where, T> operator[](unsigned i2) const {
    assert(i2 < _s2);
    return ArrayView2D<Where, T>(_s1, _s0, _data + i2 * _s1 * _s0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Where::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array3D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array3D<Where, typename std::remove_const<T>::type> clone() const { return clone_to<Where>(); }

 private:
  std::size_t _s2;
  std::size_t _s1;
  std::size_t _s0;
  T* _data;

  friend class Array3D<Where, typename std::remove_const<T>::type>;
};

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView3D<WhereFrom, T> src, ArrayView3D<WhereTo, typename std::remove_const<T>::type> dst) {
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s2() * src.s1() * src.s0(), src.data(), dst.data());
}

template<typename Where, typename T>
class Array3D : public ArrayView3D<Where, const T> {
 public:
  // RAII
  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>
  Array3D(std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView3D<Where, const T>(s2, s1, s0, Where::template alloc<T>(s2 * s1 * s0))
  {}

  template<
    typename = void,
    typename W = Where,
    typename = typename std::enable_if<W::can_be_allocated_on_device>::type
  >
  HOST_DEVICE_DECORATORS
  Array3D(std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView3D<Where, const T>(s2, s1, s0, Where::template alloc<T>(s2 * s1 * s0))
  {}

  Array3D(std::size_t s2, std::size_t s1, std::size_t s0, Zeroed) :
    ArrayView3D<Where, const T>(s2, s1, s0, Where::template alloc_zeroed<T>(s2 * s1 * s0))
  {}

  HOST_DEVICE_DECORATORS
  ~Array3D() {
    free();
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  T* data() const { return const_cast<T*>(this->_data); }

  HOST_DEVICE_DECORATORS
  ArrayView2D<Where, T> operator[](unsigned i2) const {
    assert(i2 < this->_s2);
    return ArrayView2D<Where, T>(
      this->_s1, this->_s0, data() + i2 * this->_s1 * this->_s0);
  }

  void fill_with_zeros() const {
    Where::template memreset<T>(this->total_size(), data());
  }

  // Not copyable
  Array3D(const Array3D&) = delete;
  Array3D& operator=(const Array3D&) = delete;

  // But movable
  HOST_DEVICE_DECORATORS
  Array3D(Array3D&& o) : ArrayView3D<Where, const T>(o) {
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
  }
  HOST_DEVICE_DECORATORS
  Array3D& operator=(Array3D&& o) {
    free();
    static_cast<ArrayView3D<Where, const T>&>(*this) = o;
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
    return *this;
  }

 private:
  HOST_DEVICE_DECORATORS
  void free() {
    Where::free(data());
  }
};

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView3D<Where, T> ref(const Array3D<Where, T>& a) {
  return ArrayView3D<Where, T>(a.s2(), a.s1(), a.s0(), a.data());
}

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView3D<Where, T> ref(const ArrayView3D<Where, T>& a) {
  return a;
}

template<typename WhereFrom, typename T>
template<typename WhereTo>
Array3D<WhereTo, typename std::remove_const<T>::type> ArrayView3D<WhereFrom, T>::clone_to() const {
  Array3D<WhereTo, typename std::remove_const<T>::type> dst(
    this->s2(), this->s1(), this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

template<typename Where, typename T> class Array4D;

template<typename Where, typename T>
class ArrayView4D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView4D(std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, T* data) :
    _s3(s3), _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView4D(const ArrayView4D<Where, U>& o) :
    _s3(o.s3()), _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView4D& operator=(const ArrayView4D<Where, U>& o) {
    _s3 = o.s3();
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView4D<Anywhere, U>() {
    return ArrayView4D<Anywhere, U>(_s3, _s2, _s1, _s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s3() const { return _s3; }
  HOST_DEVICE_DECORATORS
  std::size_t s2() const { return _s2; }
  HOST_DEVICE_DECORATORS
  std::size_t s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s3 * _s2 * _s1 * _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView3D<Where, T> operator[](unsigned i3) const {
    assert(i3 < _s3);
    return ArrayView3D<Where, T>(_s2, _s1, _s0, _data + i3 * _s2 * _s1 * _s0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Where::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array4D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array4D<Where, typename std::remove_const<T>::type> clone() const { return clone_to<Where>(); }

 private:
  std::size_t _s3;
  std::size_t _s2;
  std::size_t _s1;
  std::size_t _s0;
  T* _data;

  friend class Array4D<Where, typename std::remove_const<T>::type>;
};

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView4D<WhereFrom, T> src, ArrayView4D<WhereTo, typename std::remove_const<T>::type> dst) {
  assert(dst.s3() == src.s3());
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s3() * src.s2() * src.s1() * src.s0(), src.data(), dst.data());
}

template<typename Where, typename T>
class Array4D : public ArrayView4D<Where, const T> {
 public:
  // RAII
  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>
  Array4D(std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView4D<Where, const T>(s3, s2, s1, s0, Where::template alloc<T>(s3 * s2 * s1 * s0))
  {}

  template<
    typename = void,
    typename W = Where,
    typename = typename std::enable_if<W::can_be_allocated_on_device>::type
  >
  HOST_DEVICE_DECORATORS
  Array4D(std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView4D<Where, const T>(s3, s2, s1, s0, Where::template alloc<T>(s3 * s2 * s1 * s0))
  {}

  Array4D(std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Zeroed) :
    ArrayView4D<Where, const T>(s3, s2, s1, s0, Where::template alloc_zeroed<T>(s3 * s2 * s1 * s0))
  {}

  HOST_DEVICE_DECORATORS
  ~Array4D() {
    free();
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  T* data() const { return const_cast<T*>(this->_data); }

  HOST_DEVICE_DECORATORS
  ArrayView3D<Where, T> operator[](unsigned i3) const {
    assert(i3 < this->_s3);
    return ArrayView3D<Where, T>(
      this->_s2, this->_s1, this->_s0, data() + i3 * this->_s2 * this->_s1 * this->_s0);
  }

  void fill_with_zeros() const {
    Where::template memreset<T>(this->total_size(), data());
  }

  // Not copyable
  Array4D(const Array4D&) = delete;
  Array4D& operator=(const Array4D&) = delete;

  // But movable
  HOST_DEVICE_DECORATORS
  Array4D(Array4D&& o) : ArrayView4D<Where, const T>(o) {
    o._s3 = 0;
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
  }
  HOST_DEVICE_DECORATORS
  Array4D& operator=(Array4D&& o) {
    free();
    static_cast<ArrayView4D<Where, const T>&>(*this) = o;
    o._s3 = 0;
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
    return *this;
  }

 private:
  HOST_DEVICE_DECORATORS
  void free() {
    Where::free(data());
  }
};

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView4D<Where, T> ref(const Array4D<Where, T>& a) {
  return ArrayView4D<Where, T>(a.s3(), a.s2(), a.s1(), a.s0(), a.data());
}

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView4D<Where, T> ref(const ArrayView4D<Where, T>& a) {
  return a;
}

template<typename WhereFrom, typename T>
template<typename WhereTo>
Array4D<WhereTo, typename std::remove_const<T>::type> ArrayView4D<WhereFrom, T>::clone_to() const {
  Array4D<WhereTo, typename std::remove_const<T>::type> dst(
    this->s3(), this->s2(), this->s1(), this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

template<typename Where, typename T> class Array5D;

template<typename Where, typename T>
class ArrayView5D {
 public:
  // Constructor
  HOST_DEVICE_DECORATORS
  ArrayView5D(std::size_t s4, std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, T* data) :
    _s4(s4), _s3(s3), _s2(s2), _s1(s1), _s0(s0), _data(data) {}

  // No need for custom copy and move constructors and operators (cf. "Rule Of Zero" above)

  // Generalized copy constructor and operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView5D(const ArrayView5D<Where, U>& o) :
    _s4(o.s4()), _s3(o.s3()), _s2(o.s2()), _s1(o.s1()), _s0(o.s0()), _data(o.data()) {}

  template<typename U>
  HOST_DEVICE_DECORATORS
  ArrayView5D& operator=(const ArrayView5D<Where, U>& o) {
    _s4 = o.s4();
    _s3 = o.s3();
    _s2 = o.s2();
    _s1 = o.s1();
    _s0 = o.s0();
    _data = o.data();
    return *this;
  }

  // Generalized conversion operator
  template<typename U>
  HOST_DEVICE_DECORATORS
  operator ArrayView5D<Anywhere, U>() {
    return ArrayView5D<Anywhere, U>(_s4, _s3, _s2, _s1, _s0, _data);
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  std::size_t s4() const { return _s4; }
  HOST_DEVICE_DECORATORS
  std::size_t s3() const { return _s3; }
  HOST_DEVICE_DECORATORS
  std::size_t s2() const { return _s2; }
  HOST_DEVICE_DECORATORS
  std::size_t s1() const { return _s1; }
  HOST_DEVICE_DECORATORS
  std::size_t s0() const { return _s0; }

  HOST_DEVICE_DECORATORS
  std::size_t total_size() const { return _s4 * _s3 * _s2 * _s1 * _s0; }

  HOST_DEVICE_DECORATORS
  ArrayView4D<Where, T> operator[](unsigned i4) const {
    assert(i4 < _s4);
    return ArrayView4D<Where, T>(_s3, _s2, _s1, _s0, _data + i4 * _s3 * _s2 * _s1 * _s0);
  }

  HOST_DEVICE_DECORATORS
  T* data() const { return _data; }

  void fill_with_zeros() const {
    Where::template memreset<T>(total_size(), data());
  }

  // Clonable
  template<typename WhereTo>
  Array5D<WhereTo, typename std::remove_const<T>::type> clone_to() const;

  Array5D<Where, typename std::remove_const<T>::type> clone() const { return clone_to<Where>(); }

 private:
  std::size_t _s4;
  std::size_t _s3;
  std::size_t _s2;
  std::size_t _s1;
  std::size_t _s0;
  T* _data;

  friend class Array5D<Where, typename std::remove_const<T>::type>;
};

template<typename WhereFrom, typename WhereTo, typename T>
void copy(ArrayView5D<WhereFrom, T> src, ArrayView5D<WhereTo, typename std::remove_const<T>::type> dst) {
  assert(dst.s4() == src.s4());
  assert(dst.s3() == src.s3());
  assert(dst.s2() == src.s2());
  assert(dst.s1() == src.s1());
  assert(dst.s0() == src.s0());

  From<WhereFrom>::template To<WhereTo>::template copy(
    src.s4() * src.s3() * src.s2() * src.s1() * src.s0(), src.data(), dst.data());
}

template<typename Where, typename T>
class Array5D : public ArrayView5D<Where, const T> {
 public:
  // RAII
  template<typename W = Where, typename = typename std::enable_if<!W::can_be_allocated_on_device>::type>
  Array5D(std::size_t s4, std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView5D<Where, const T>(s4, s3, s2, s1, s0, Where::template alloc<T>(s4 * s3 * s2 * s1 * s0))
  {}

  template<
    typename = void,
    typename W = Where,
    typename = typename std::enable_if<W::can_be_allocated_on_device>::type
  >
  HOST_DEVICE_DECORATORS
  Array5D(std::size_t s4, std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Uninitialized) :
    ArrayView5D<Where, const T>(s4, s3, s2, s1, s0, Where::template alloc<T>(s4 * s3 * s2 * s1 * s0))
  {}

  Array5D(std::size_t s4, std::size_t s3, std::size_t s2, std::size_t s1, std::size_t s0, Zeroed) :
    ArrayView5D<Where, const T>(s4, s3, s2, s1, s0, Where::template alloc_zeroed<T>(s4 * s3 * s2 * s1 * s0))
  {}

  HOST_DEVICE_DECORATORS
  ~Array5D() {
    free();
  }

  // Accessors
  HOST_DEVICE_DECORATORS
  T* data() const { return const_cast<T*>(this->_data); }

  HOST_DEVICE_DECORATORS
  ArrayView4D<Where, T> operator[](unsigned i4) const {
    assert(i4 < this->_s4);
    return ArrayView4D<Where, T>(
      this->_s3, this->_s2, this->_s1, this->_s0, data() + i4 * this->_s3 * this->_s2 * this->_s1 * this->_s0);
  }

  void fill_with_zeros() const {
    Where::template memreset<T>(this->total_size(), data());
  }

  // Not copyable
  Array5D(const Array5D&) = delete;
  Array5D& operator=(const Array5D&) = delete;

  // But movable
  HOST_DEVICE_DECORATORS
  Array5D(Array5D&& o) : ArrayView5D<Where, const T>(o) {
    o._s4 = 0;
    o._s3 = 0;
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
  }
  HOST_DEVICE_DECORATORS
  Array5D& operator=(Array5D&& o) {
    free();
    static_cast<ArrayView5D<Where, const T>&>(*this) = o;
    o._s4 = 0;
    o._s3 = 0;
    o._s2 = 0;
    o._s1 = 0;
    o._s0 = 0;
    o._data = nullptr;
    return *this;
  }

 private:
  HOST_DEVICE_DECORATORS
  void free() {
    Where::free(data());
  }
};

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView5D<Where, T> ref(const Array5D<Where, T>& a) {
  return ArrayView5D<Where, T>(a.s4(), a.s3(), a.s2(), a.s1(), a.s0(), a.data());
}

template<typename Where, typename T>
HOST_DEVICE_DECORATORS
ArrayView5D<Where, T> ref(const ArrayView5D<Where, T>& a) {
  return a;
}

template<typename WhereFrom, typename T>
template<typename WhereTo>
Array5D<WhereTo, typename std::remove_const<T>::type> ArrayView5D<WhereFrom, T>::clone_to() const {
  Array5D<WhereTo, typename std::remove_const<T>::type> dst(
    this->s4(), this->s3(), this->s2(), this->s1(), this->s0(), uninitialized);
  copy(*this, ref(dst));  // NOLINT(build/include_what_you_use)
  return dst;
}

// END GENERATED SECTION: arrays-and-array-views


/*                          *
 * Grid and block utilities *
 *                          */

#ifdef LINCS_HAS_NVCC

struct Grid {
  const dim3 gridDim;
  const dim3 blockDim;
};

#define LOVE_CONFIG(grid) grid.gridDim, grid.blockDim

template<unsigned BLOCKDIM_X>
struct GridFactory1D {
  HOST_DEVICE_DECORATORS
  static Grid make(unsigned x) {
    return Grid {
      {
        (x + BLOCKDIM_X - 1) / BLOCKDIM_X,
        1,
        1
      },
      blockDim,
    };
  }

  HOST_DEVICE_DECORATORS
  static Grid fixed(unsigned x) {
    return Grid {
      {x, 1, 1},
      blockDim,
    };
  }

#ifdef __NVCC__
  __device__ static unsigned x() {
    assert(::blockDim.x == BLOCKDIM_X);
    return blockIdx.x * BLOCKDIM_X + threadIdx.x;
  }
#endif  // __NVCC__

  static constexpr dim3 blockDim = {BLOCKDIM_X, 1, 1};
};

template<unsigned BLOCKDIM_X, unsigned BLOCKDIM_Y>
struct GridFactory2D {
  HOST_DEVICE_DECORATORS
  static Grid make(unsigned x, unsigned y) {
    return Grid {
      {
        (x + BLOCKDIM_X - 1) / BLOCKDIM_X,
        (y + BLOCKDIM_Y - 1) / BLOCKDIM_Y,
        1
      },
      blockDim,
    };
  }

  HOST_DEVICE_DECORATORS
  static Grid fixed(unsigned x, unsigned y) {
    return Grid {
      {x, y, 1},
      blockDim,
    };
  }

#ifdef __NVCC__
  __device__ static unsigned x() {
    assert(::blockDim.x == BLOCKDIM_X);
    return blockIdx.x * BLOCKDIM_X + threadIdx.x;
  }

  __device__ static unsigned y() {
    assert(::blockDim.y == BLOCKDIM_Y);
    return blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  }
#endif  // __NVCC__

  static constexpr dim3 blockDim = {BLOCKDIM_X, BLOCKDIM_Y, 1};
};

template<unsigned BLOCKDIM_X, unsigned BLOCKDIM_Y, unsigned BLOCKDIM_Z>
struct GridFactory3D {
  HOST_DEVICE_DECORATORS
  static Grid make(unsigned x, unsigned y, unsigned z) {
    return Grid {
      {
        (x + BLOCKDIM_X - 1) / BLOCKDIM_X,
        (y + BLOCKDIM_Y - 1) / BLOCKDIM_Y,
        (z + BLOCKDIM_Z - 1) / BLOCKDIM_Z
      },
      blockDim,
    };
  }

  HOST_DEVICE_DECORATORS
  static Grid fixed(unsigned x, unsigned y, unsigned z) {
    return Grid {
      {x, y, z},
      blockDim,
    };
  }

#ifdef __NVCC__
  __device__ static unsigned x() {
    assert(::blockDim.x == BLOCKDIM_X);
    return blockIdx.x * BLOCKDIM_X + threadIdx.x;
  }

  __device__ static unsigned y() {
    assert(::blockDim.y == BLOCKDIM_Y);
    return blockIdx.y * BLOCKDIM_Y + threadIdx.y;
  }

  __device__ static unsigned z() {
    assert(::blockDim.z == BLOCKDIM_Z);
    return blockIdx.z * BLOCKDIM_Z + threadIdx.z;
  }
#endif  // __NVCC__

  static constexpr dim3 blockDim = {BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z};
};

#endif  // LINCS_HAS_NVCC

#undef HOST_DEVICE_DECORATORS

#endif  // LOV_E_HPP_
