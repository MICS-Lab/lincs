// Copyright 2021-2022 Vincent Jacques

#ifndef CUDA_UTILS_HPP_
#define CUDA_UTILS_HPP_

#include <cuda_runtime.h>

#include <cassert>
#include <cstdio>


#ifdef __NVCC__
#define HOST_DEVICE_DECORATORS __host__ __device__
#else
#define HOST_DEVICE_DECORATORS
#endif

HOST_DEVICE_DECORATORS
inline void checkCudaErrors_(const char* file, unsigned int line) {
  cudaError_t error = cudaGetLastError();
  if (error) {
    printf(
      "CUDA ERROR, detected at %s:%i: %i %s\n",
      file, line, static_cast<unsigned int>(error), cudaGetErrorName(error));
    #ifndef __CUDA_ARCH__
    assert(false);  // Dump core for further investigations
    #endif
  }
}

#define checkCudaErrors() checkCudaErrors_(__FILE__, __LINE__)

template<typename T>
T* alloc_host(unsigned int n) {
  if (n == 0) return nullptr;
  T* p;
  cudaMallocHost(&p, n * sizeof(T));
  checkCudaErrors();
  return p;
}

template<typename T>
void free_host(T* p) {
  if (p == nullptr) return;
  cudaFreeHost(p);
  checkCudaErrors();
}

template<typename T>
HOST_DEVICE_DECORATORS
T* alloc_device(unsigned int n) {
  if (n == 0) return nullptr;
  T* p;
  cudaMalloc(&p, n * sizeof(T));
  checkCudaErrors();
  return p;
}

template<typename T>
HOST_DEVICE_DECORATORS
void free_device(T* p) {
  if (p == nullptr) return;
  cudaFree(p);
  checkCudaErrors();
}

template<typename T>
void copy_host_to_device(unsigned int n, const T* src, T* dst) {
  if (n == 0) return;
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
  checkCudaErrors();
}

template<typename T>
T* clone_host_to_device(unsigned int n, const T* src) {
  T* dst = alloc_device<T>(n);
  copy_host_to_device(n, src, dst);
  return dst;
}

template<typename T>
void copy_device_to_host(unsigned int n, const T* src, T* dst) {
  if (n == 0) return;
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
  checkCudaErrors();
}

template<typename T>
T* clone_device_to_host(unsigned int n, const T* src) {
  T* dst = alloc_host<T>(n);
  copy_device_to_host(n, src, dst);
  return dst;
}

struct Host {
  template<typename T>
  static T* alloc(unsigned int n) { return alloc_host<T>(n); }

  template<typename T>
  static void free(T* p) { free_host(p); }
};

struct Device {
  template<typename T>
  static T* alloc(unsigned int n) { return alloc_device<T>(n); }

  template<typename T>
  static void free(T* p) { free_device(p); }
};

template<typename From, typename To>
struct FromTo;

template<>
struct FromTo<Host, Device> {
  template<typename T>
  static T* clone(unsigned int n, T* p) {
    return clone_host_to_device(n, p);
  }
};

template<>
struct FromTo<Device, Host> {
  template<typename T>
  static T* clone(unsigned int n, T* p) {
    return clone_device_to_host(n, p);
  }
};

#define BLOCKDIM 512
#define CONFIG(size) (size) / BLOCKDIM + ((size) % BLOCKDIM ? 1 : 0), BLOCKDIM

#endif  // CUDA_UTILS_HPP_
