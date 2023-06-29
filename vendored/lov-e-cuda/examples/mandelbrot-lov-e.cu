// Copyright 2014 Andy Adinets
// Copyright 2022 Vincent Jacques
// Copyright 2022 Laurent Cabaret
// File copied from https://github.com/canonizer/mandelbrot-dyn
// (https://developer.nvidia.com/blog/introduction-cuda-dynamic-parallelism/)
// then modified by Vincent Jacques to fit this project's coding guidelines

#include <png.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../lov-e.hpp"


#define H (16 * 1024)
#define W (16 * 1024)
#define IMAGE_PATH "./mandelbrot.png"
#define MAX_DWELL 512
#define CUT_DWELL (MAX_DWELL / 4)
// Block size along
#define BSX 64
#define BSY 4


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


void save_image(const char* filename, ArrayView2D<Host, const int> dwells) {
  const unsigned h = dwells.s1();
  const unsigned w = dwells.s0();

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
      dwell_color(&r, &g, &b, dwells[y][x]);
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


// Compute the dwell for a single pixel
__device__
int pixel_dwell(int w, int h, complex cmin, complex cmax, int x, int y) {
  const complex dc = cmax - cmin;
  const float fx = static_cast<float>(x) / w;
  const float fy = static_cast<float>(y) / h;
  const complex c = cmin + complex(fx * dc.re, fy * dc.im);
  int dwell = 0;
  complex z = c;
  while (dwell < MAX_DWELL && abs2(z) < 2 * 2) {
    z = z * z + c;
    dwell++;
  }
  return dwell;
}

typedef GridFactory2D<BSX, BSY> grid;

/*
Compute the dwells for Mandelbrot image

@param dwells the output array
@param w the width of the output image
@param h the height of the output image
@param cmin the complex value associated with the bottom-left corner of the image
@param cmax the complex value associated with the top-right corner of the image
*/
__global__
void mandelbrot_k(ArrayView2D<Device, int> dwells, complex cmin, complex cmax) {
  const unsigned h = dwells.s1();
  const unsigned w = dwells.s0();
  const int x = grid::x();
  const int y = grid::y();
  dwells[y][x] = pixel_dwell(w, h, cmin, cmax, x, y);
}

int main(int, char*[]) {
  const int w = W;
  const int h = H;

  Array2D<Device, int> d_dwells(h, w, uninitialized);

  const Grid grid = grid::make(w, h);

  const double t1 = omp_get_wtime();
  mandelbrot_k<<<LOVE_CONFIG(grid)>>>(ref(d_dwells), complex(-1.5, -1), complex(0.5, 1));
  check_last_cuda_error_sync_device();

  const double t2 = omp_get_wtime();

  Array2D<Host, int> h_dwells = d_dwells.clone_to<Host>();
  save_image(IMAGE_PATH, h_dwells);

  const double gpu_time = t2 - t1;
  printf("Mandelbrot set computed in %.3lf s, at %.3lf Mpix/s\n", gpu_time, h * w * 1e-6 / gpu_time);
}
