#pragma once

#include <x86intrin.h>

// Serial version
void saxpy_serial(int N, float scale, float X[], float Y[], float result[]);

// AVX 1.0 version
#ifdef __AVX__
void saxpy_avx(int N, float scale, float X[], float Y[], float result[]);
#endif

// AVX 2.0 version
#ifdef __AVX2__
void saxpy_avx2(int N, float scale, float X[], float Y[], float result[]);
#endif

#ifdef __AVX512F__
void saxpy_avx512(int N, float scale, float X[], float Y[], float result[]);
#endif
