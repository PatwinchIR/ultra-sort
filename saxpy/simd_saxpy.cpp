#include "simd_saxpy.h"
#include <stdio.h>


void saxpy_serial(int N, float scale, float X[], float Y[], float result[]) {
  for(int i = 0; i < N; i++) {
    result[i] = scale*X[i] + Y[i];
  }
}

#ifdef __AVX__
void saxpy_avx(int N, float scale, float X[], float Y[], float result[]) {
  float scale_arr[4] = {scale, scale, scale, scale};
  // Cast 'scale' to __m128
  __m128* scaleVec = (__m128*)scale_arr;
  __m128* XVec = (__m128*)X;
  __m128* YVec = (__m128*)Y;
  int vecWidth = 4;
  for(int i = 0; i < N; i+=vecWidth) {
    // a*X
    __m128 ax = _mm_mul_ps(*XVec, *scaleVec);
    // aX + Y
    __m128 res = _mm_add_ps(ax, *YVec);
    // Do a Stream store(direct write to memory)
    _mm_storeu_ps(&result[i], res);
    // Increment to work on next 4 numbers
    XVec++;
    YVec++;
  }
}
#endif

#ifdef __AVX2__
void saxpy_avx2(int N, float scale, float X[], float Y[], float result[]) {
  float scale_arr[8] = {scale, scale, scale, scale,
                        scale, scale, scale, scale};
  // Cast 'scale' to __m128
  __m256* scaleVec = (__m256*)scale_arr;
  __m256* XVec = (__m256*)X;
  __m256* YVec = (__m256*)Y;
  int vecWidth = 8;
  for(int i = 0; i < N; i+=vecWidth) {
    // a*X
    __m256 ax = _mm256_mul_ps(*XVec, *scaleVec);
    // aX + Y
    __m256 res = _mm256_add_ps(ax, *YVec);
    // Do a store
    _mm256_store_ps(&result[i], res);
    // Increment to work on next 8 numbers
    XVec++;
    YVec++;
  }
}
#endif

#ifdef __AVX512F__
void saxpy_avx512(int N, float scale, float X[], float Y[], float result[]) {
  float scale_arr[16] = {scale, scale, scale, scale,
                         scale, scale, scale, scale,
                         scale, scale, scale, scale,
                         scale, scale, scale, scale};
  // Cast 'scale' to __m128
  __m512* scaleVec = (__m512*)scale_arr;
  __m512* XVec = (__m512*)X;
  __m512* YVec = (__m512*)Y;
  int vecWidth = 16;
  for(int i = 0; i < N; i+=vecWidth) {
    // a*X
    __m512 ax = _mm512_mul_ps(*XVec, *scaleVec);
    // aX + Y
    __m512 res = _mm512_add_ps(ax, *YVec);
    // Do a store
    _mm512_store_ps(&result[i], res);
    // Increment to work on next 16 numbers
    XVec++;
    YVec++;
  }
}
#endif
