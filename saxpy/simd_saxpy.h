#pragma once

#include <emmintrin.h> // For _mm_mul_ps

// Serial version
void saxpy_serial(int N, float scale, float X[], float Y[], float result[]);

// AVX 1.0 version
#ifdef __AVX__
void saxpy_avx(int N, float scale, float X[], float Y[], float result[]);
#endif