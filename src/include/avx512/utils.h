#pragma once

#include "common.h"

#ifdef AVX512

class AVX512Util{
 public:
  static void LoadReg(__m512i &r, int* arr);
  static void StoreReg(const __m512i &r, int* arr);
  static void LoadReg(__m512i &r, int64_t* arr);
  static void StoreReg(const __m512i &r, int64_t* arr);
  static void MinMax8(__m512i &a, __m512i &b);
  static void MinMax16(__m512i &a, __m512i &b);
  static void BitonicSort8x8(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
                             __m512i &r4, __m512i &r5, __m512i &r6, __m512i &r7);
  static void BitonicSort16x16(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
                               __m512i &r4, __m512i &r5, __m512i &r6, __m512i &r7,
                               __m512i &r8, __m512i &r9, __m512i &r10, __m512i &r11,
                               __m512i &r12, __m512i &r13, __m512i &r14, __m512i &r15);
  static void Transpose8x8(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                           __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7);
  static void Trowanspose16x16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                               __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
                               __m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11,
                               __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15);
  static __m512i Reverse8(__m512i& v);
  static __m512i Reverse16(__m512i& v);
  static void MinMax8(const __m512i& a, const __m512i& b,
                     __m512i& minab, __m512i& maxab);
  static void IntraRegisterSort8x8(__m512i& a8, __m512i& b8);
  static void IntraRegisterSort16x16(__m512i& a4, __m512i& b4);
  static void BitonicMerge8(__m512i& a, __m512i& b);
  static void BitonicMerge16(__m512i& a, __m512i& b);
};

#endif