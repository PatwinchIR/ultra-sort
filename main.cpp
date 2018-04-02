#include <iostream>
#include <cassert>
#include "metrics/cycletimer.h"
#include "saxpy/simd_saxpy.h"


void initialize_saxpy(float* &X, float* &Y, float* &result, int N) {
  X = new float[N];
  Y = new float[N];
  result = new float[N];
  for(int i = 0; i < N; i++) {
    X[i] = i;
    Y[i] = i;
  }
}

void initialize_result(float* &result, int N) {
  result = new float[N];
}

void check_correctness(const float* true_arr, const float* exp_arr, int N) {
  for (int i = 0; i < N; ++i) {
    assert(true_arr[i] == exp_arr[i]);
  }
}

int main() {
  // Initialization
  int N = 10000;
  float scale = 1.5;
  float *X;
  float *Y;
  float *result_serial;
  float* result_avx;
//  double start, end;
  initialize_saxpy(X, Y, result_serial, N);

  // Serial Saxpy
  // TODO: Below throws a linker error - why??
   currentSeconds();
//  start = currentSeconds();
  saxpy_serial(N, scale, X, Y, result_serial);
//  end = currentSeconds();
//  printf("[Saxpy Serial] %d elements: %.5f seconds", N, end - start);

#ifdef __AVX__
  initialize_result(result_avx, N);
//  start = currentSeconds();
  saxpy_avx(N, scale, X, Y, result_avx);
//  end = currentSeconds();
  check_correctness(result_serial, result_avx, N);
//  printf("[Saxpy AVX] %d elements: %.5f seconds", N, end - start);
#else
  std::cout << "Missing AVX instructions" << std::endl;
#endif
  return 0;
}
