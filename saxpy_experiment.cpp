#include <iostream>
#include <cassert>
#include "metrics/cycletimer.h"
#include "saxpy/simd_saxpy.h"
#include "common.h"

void initialize_saxpy(float* &X, float* &Y, float* &result, int N) {
  aligned_init<float>(X, N);
  aligned_init<float>(Y, N);
  aligned_init<float>(result, N);
  for(int i = 0; i < N; i++) {
    X[i] = i;
    Y[i] = i;
  }
}

void check_correctness(const float* true_arr, const float* exp_arr, int N) {
  for (int i = 0; i < N; ++i) {
    assert(true_arr[i] == exp_arr[i]);
  }
}

int main() {
  // Initialization
  int N = 65536;
  float scale = 1.5;
  float *X;
  float *Y;
  float *result_serial;
  float* result_avx;
  double start, end;
  initialize_saxpy(X, Y, result_serial, N);

  // Serial Saxpy
  start = currentSeconds();
  saxpy_serial(N, scale, X, Y, result_serial);
  end = currentSeconds();
  printf("[Saxpy Serial] %d elements: %.8f seconds\n", N, end - start);

#ifdef __AVX__
  aligned_init<float>(result_avx, N);
  start = currentSeconds();
  saxpy_avx(N, scale, X, Y, result_avx);
  end = currentSeconds();
  check_correctness(result_serial, result_avx, N);
  delete result_avx;
  printf("[Saxpy AVX] %d elements: %.8f seconds\n", N, end - start);
#else
  printf("Missing AVX instructions\n");
#endif

#ifdef __AVX2__
  aligned_init<float>(result_avx, N);
  start = currentSeconds();
  saxpy_avx2(N, scale, X, Y, result_avx);
  end = currentSeconds();
  check_correctness(result_serial, result_avx, N);
  delete result_avx;
  printf("[Saxpy AVX2] %d elements: %.8f seconds\n", N, end - start);
#else
  printf("Missing AVX2 instructions\n");
#endif

#ifdef __AVX512F__
  aligned_init<float>(result_avx, N);
  start = currentSeconds();
  saxpy_avx512(N, scale, X, Y, result_avx);
  end = currentSeconds();
  check_correctness(result_serial, result_avx, N);
  delete result_avx;
  printf("[Saxpy AVX512] %d elements: %.8f seconds\n", N, end - start);
#else
  printf("Missing AVX512 instructions\n");
#endif
  return 0;
}
