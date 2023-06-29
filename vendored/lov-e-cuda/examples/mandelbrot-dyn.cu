// Copyright 2014 Andy Adinets
// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret
// File copied from https://github.com/canonizer/mandelbrot-dyn
// (https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)
// then modified by Vincent Jacques to fit this project's coding guidelines

#include <assert.h>
#include <png.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


#define H (16 * 1024)
#define W (16 * 1024)
#define IMAGE_PATH "./mandelbrot.png"
#define MAX_DWELL 512
#define CUT_DWELL (MAX_DWELL / 4)
// Block size along
#define BSX 64
#define BSY 4
// Maximum recursion depth
#define MAX_DEPTH 4
// Region below which do per-pixel
#define MIN_SIZE 32
// Subdivision factor along each axis
#define SUBDIV 4
// Subdivision when launched from host
#define INIT_SUBDIV 32


// Get the color, given the dwell (on host)
void dwell_color(int* r, int* g, int* b, int dwell) {
  // Black for the Mandelbrot set
  if (dwell >= MAX_DWELL) {
    *r = *g = *b = 0;
  } else {
    // Cut at zero
    if (dwell < 0)
      dwell = 0;
    if (dwell <= CUT_DWELL) {
      // From black to blue the first half
      *r = *g = 0;
      *b = 128 + dwell * 127 / (CUT_DWELL);
    } else {
      // From blue to white for the second half
      *b = 255;
      *r = *g = (dwell - CUT_DWELL) * 255 / (MAX_DWELL - CUT_DWELL);
    }
  }
}


void save_image(const char* filename, int* dwells, unsigned w, unsigned h) {
  // Code taken from http://www.labbookpages.co.uk/software/imgProc/libPNG.html
  png_bytep row;
  FILE* fp = fopen(filename, "wb");
  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_init_io(png_ptr, fp);
  png_set_IHDR(
    png_ptr, info_ptr, w, h, 8, PNG_COLOR_TYPE_RGB,
    PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
  png_text title_text;
  title_text.compression = PNG_TEXT_COMPRESSION_NONE;
  title_text.key = const_cast<char*>("Title");
  title_text.text = const_cast<char*>("Mandelbrot set, per-pixel");
  png_set_text(png_ptr, info_ptr, &title_text, 1);
  png_write_info(png_ptr, info_ptr);
  row = reinterpret_cast<png_bytep>(malloc(3 * w * sizeof(png_byte)));
  for (unsigned y = 0; y < h; y++) {
    for (unsigned x = 0; x < w; x++) {
      int r, g, b;
      dwell_color(&r, &g, &b, dwells[y * w + x]);
      row[3 * x + 0] = (png_byte)r;
      row[3 * x + 1] = (png_byte)g;
      row[3 * x + 2] = (png_byte)b;
    }
    png_write_row(png_ptr, row);
  }
  png_write_end(png_ptr, nullptr);

  fclose(fp);
  png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
  png_destroy_write_struct(&png_ptr, nullptr);
  free(row);
}


struct complex {
  __host__ __device__ complex(const float re_, const float im_ = 0) : re(re_), im(im_) {}

  float re, im;
};

inline __host__ __device__ complex operator+(const complex &a, const complex &b) {
  return complex(a.re + b.re, a.im + b.im);
}

inline __host__ __device__ complex operator-(const complex &a) {
  return complex(-a.re, -a.im);
}

inline __host__ __device__ complex operator-(const complex &a, const complex &b) {
  return complex(a.re - b.re, a.im - b.im);
}

inline __host__ __device__ complex operator*(const complex &a, const complex &b) {
  return complex(a.re * b.re - a.im * b.im, a.im * b.re + a.re * b.im);
}

inline __host__ __device__ float abs2(const complex &a) {
  return a.re * a.re + a.im * a.im;
}

inline __host__ __device__ complex operator/(const complex &a, const complex &b) {
  const float invabs2 = 1 / abs2(b);
  return complex((a.re * b.re + a.im * b.im) * invabs2,
                 (a.im * b.re - b.im * a.re) * invabs2);
}


// Check CUDA errors
#define cucheck(call) { \
  cudaError_t res = (call); \
  if (res != cudaSuccess) { \
    const char* err_str = cudaGetErrorString(res); \
    fprintf(stderr, "%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call); \
    exit(-1); \
  } \
}

#define cucheck_dev(call) { \
  cudaError_t res = (call); \
  if (res != cudaSuccess) { \
    const char* err_str = cudaGetErrorString(res); \
    printf("%s (%d): %s in %s", __FILE__, __LINE__, err_str, #call); \
    assert(0); \
  } \
}


__host__ __device__
int divup(int x, int y) { return x / y + (x % y ? 1 : 0); }

// Compute the dwell for a single pixel
__device__
int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y) {
  const complex dc = cmax - cmin;
  const float fx = static_cast<float>(x) / w, fy = static_cast<float>(y) / h;
  const complex c = cmin + complex(fx * dc.re, fy * dc.im);
  int dwell = 0;
  complex z = c;
  while (dwell < MAX_DWELL && abs2(z) < 2 * 2) {
    z = z * z + c;
    dwell++;
  }
  return dwell;
}

// Binary operation for common dwell "reduction": MAX_DWELL + 1 = neutral element, -1 = dwells are different
#define NEUT_DWELL (MAX_DWELL + 1)
#define DIFF_DWELL (-1)
__device__
int same_dwell(int d1, int d2) {
  if (d1 == d2) {
    return d1;
  } else if (d1 == NEUT_DWELL || d2 == NEUT_DWELL) {
    return min(d1, d2);
  } else {
    return DIFF_DWELL;
  }
}

// Evaluate the common border dwell, if it exists
__device__
int border_dwell(int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
  // Check whether all boundary pixels have the same dwell
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int bs = blockDim.x * blockDim.y;
  int comm_dwell = NEUT_DWELL;
  // For all boundary pixels, distributed across threads
  for (int r = tid; r < d; r += bs) {
    // For each boundary: b = 0 is east, then counter-clockwise
    for (int b = 0; b < 4; b++) {
      const int x = b % 2 != 0 ? x0 + r : (b == 0 ? x0 + d - 1 : x0);
      const int y = b % 2 == 0 ? y0 + r : (b == 1 ? y0 + d - 1 : y0);
      const int dwell = pixel_dwell(w, h, cmin, cmax, x, y);
      comm_dwell = same_dwell(comm_dwell, dwell);
    }
  }
  // Reduce across threads in the block
  __shared__ int ldwells[BSX * BSY];
  int nt = min(d, BSX * BSY);
  if (tid < nt) {
    ldwells[tid] = comm_dwell;
  }
  __syncthreads();
  for (; nt > 1; nt /= 2) {
    if (tid < nt / 2) {
      ldwells[tid] = same_dwell(ldwells[tid], ldwells[tid + nt / 2]);
    }
    __syncthreads();
  }
  return ldwells[0];
}

// Fill the image region with a specific dwell value
__global__
void dwell_fill_k(int* dwells, int w, int x0, int y0, int d, int dwell) {
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (x < d && y < d) {
    x += x0, y += y0;
    dwells[y * w + x] = dwell;
  }
}

// Fill in per-pixel values of the portion of the Mandelbrot set
__global__
void mandelbrot_pixel_k(int* dwells, int w, int h, complex cmin, complex cmax, int x0, int y0, int d) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (x < d && y < d) {
    x += x0, y += y0;
    dwells[y * w + x] = pixel_dwell(w, h, cmin, cmax, x, y);
  }
}

/*
Compute the dwells for Mandelbrot image using dynamic parallelism

One block is launched per pixel. The algorithm reverts to per-pixel Mandelbrot evaluation
once either maximum depth or minimum size is reached.

@param dwells the output array
@param w the width of the output image
@param h the height of the output image
@param cmin the complex value associated with the left-bottom corner of the image
@param cmax the complex value associated with the right-top corner of the image
@param x0 the starting x coordinate of the portion to compute
@param y0 the starting y coordinate of the portion to compute
@param d the size of the portion to compute (the portion is always a square)
@param depth kernel invocation depth
*/
__global__
void mandelbrot_block_k(
  int* dwells, int w, int h, complex cmin, complex cmax, int x0, int y0,  int d, int depth
) {
  x0 += d * blockIdx.x, y0 += d * blockIdx.y;
  int comm_dwell = border_dwell(w, h, cmin, cmax, x0, y0, d);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    if (comm_dwell != DIFF_DWELL) {
      // Uniform dwell, just fill
      const dim3 threads(BSX, BSY);
      const dim3 blocks(divup(d, BSX), divup(d, BSY));
      dwell_fill_k<<<blocks, threads>>>(dwells, w, x0, y0, d, comm_dwell);
    } else if (depth + 1 < MAX_DEPTH && d / SUBDIV > MIN_SIZE) {
      // Subdivide recursively
      const dim3 threads(BSX, BSY);
      const dim3 blocks(SUBDIV, SUBDIV);
      mandelbrot_block_k<<<blocks, threads>>>(dwells, w, h, cmin, cmax, x0, y0, d / SUBDIV, depth+ 1);
    } else {
      // Leaf: per-pixel kernel
      const dim3 threads(BSX, BSY);
      const dim3 blocks(divup(d, BSX), divup(d, BSY));
      mandelbrot_pixel_k<<<blocks, threads>>>(dwells, w, h, cmin, cmax, x0, y0, d);
    }
    cucheck_dev(cudaGetLastError());
  }
}

int main(int, char*[]) {
  const int w = W;
  const int h = H;
  const size_t dwell_sz = w * h * sizeof(int);

  int* d_dwells;
  cucheck(cudaMalloc(reinterpret_cast<void**>(&d_dwells), dwell_sz));
  int* const h_dwells = reinterpret_cast<int*>(malloc(dwell_sz));

  const dim3 threads(BSX, BSY);
  const dim3 blocks(INIT_SUBDIV, INIT_SUBDIV);

  const double t1 = omp_get_wtime();
  mandelbrot_block_k<<<blocks, threads>>>(
    d_dwells, w, h, complex(-1.5, -1), complex(0.5, 1), 0, 0, w / INIT_SUBDIV, 1);
  cucheck(cudaDeviceSynchronize());
  const double t2 = omp_get_wtime();

  cucheck(cudaMemcpy(h_dwells, d_dwells, dwell_sz, cudaMemcpyDeviceToHost));
  save_image(IMAGE_PATH, h_dwells, w, h);

  const double gpu_time = t2 - t1;
  printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);

  free(h_dwells);
  cudaFree(d_dwells);
}
