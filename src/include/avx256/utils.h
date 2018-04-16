#pragma once

#include "common.h"

#ifdef AVX2

class AVX256Util{
 public:
  static void LoadReg(__m256i &r, const int* arr);
  static void StoreReg(const __m256i &r, int* arr);
  static void MinMax8(__m256i &b, __m256i &a);
  static void BitonicSort8x8(__m256i &r0, __m256i &r1, __m256i &r2, __m256i &r3,
                             __m256i &r4, __m256i &r5, __m256i &r6, __m256i &r7);
  static void Transpose8x8(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                           __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7);
  static __m256i Reverse8(__m256i& v);
  static void BitonicMerge8(__m256i& a, __m256i& b);
  static void MinMax8(const __m256i& a, const __m256i& b,
                     __m256i& minab, __m256i& maxab);
 private:
  static void IntraRegisterSort8x8(__m256i& a8, __m256i& b8);
};

#endif