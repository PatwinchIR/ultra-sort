#include "simd_saxpy.h"


void saxpy_serial(int N, float scale, float X[], float Y[], float result[]) {
  for(int i = 0; i < N; i++) {
    result[i] = scale*X[i] + Y[i];
  }
}

#ifdef __AVX__
void saxpy_avx(int N, float scale, float X[], float Y[], float result[]) {
  float aVec[4] = {scale, scale, scale, scale};
  // Cast 'scale' to __m128
  __m128* aVecMem = (__m128*)aVec;
  __m128* scaleVec = aVecMem;
  // Setup X/Y as __m128i(to allow for streaming load)
  __m128* XVec = (__m128*)X;
  __m128* YVec = (__m128*)Y;
  int vecWidth = 4;
  for(int i = 0; i < N; i+=vecWidth) {
    // a*X
    __m128 ax = _mm_mul_ps(*XVec, *scaleVec);
    // aX + Y
    __m128 res = _mm_add_ps(ax, *YVec);
    // Do a Stream store(direct write to memory)
    _mm_store_ps(&result[i], res);
    // Increment to work on next 4 numbers
    XVec++;
    YVec++;
  }
}
#endif
