// Copyright 2021 Vincent Jacques

#ifndef CUDA_UTILS_HPP_
#define CUDA_UTILS_HPP_


template<typename T>
T* alloc_host(int n) {
  T* p;
  cudaMallocHost(&p, n * sizeof(T));
  return p;
}

template<typename T>
void free_host(T* p) {
  cudaFreeHost(p);
}

template<typename T>
T* alloc_device(int n) {
  T* p;
  cudaMalloc(&p, n * sizeof(T));
  return p;
}

template<typename T>
void free_device(T* p) {
  cudaFree(p);
}

template<typename T>
void copy_host_to_device(int n, const T* src, T* dst) {
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
T* clone_host_to_device(int n, const T* src) {
  T* dst = alloc_device<T>(n);
  copy_host_to_device(n, src, dst);
  return dst;
}

template<typename T>
void copy_device_to_host(int n, const T* src, T* dst) {
  cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
T* clone_device_to_host(int n, const T* src) {
  T* dst = alloc_host<T>(n);
  copy_device_to_host(n, src, dst);
  return dst;
}

#endif  // CUDA_UTILS_HPP_
