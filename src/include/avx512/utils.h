#pragma once

#include "common.h"

#ifdef AVX512

class AVX512Util{
 public:
  // Load/Stores
  template <typename InType, typename RegType>
  static void LoadReg(RegType &r, InType* arr);
  template <typename InType, typename RegType>
  static void StoreReg(const RegType &r, InType* arr);

//  // Converters
//  static __m512d Int64ToDoubleReg(const __m512i &repi64);
//  static __m512i DoubleToInt64Reg(const __m512d &rd);

  // Min/Max
  static void MinMax16(__m512i &a, __m512i &b);
  static void MinMax16(const __m512i& a, const __m512i& b,
                      __m512i& minab, __m512i& maxab);
  static void MinMax16(__m512 &a, __m512 &b);
  static void MinMax16(const __m512& a, const __m512& b,
                      __m512& minab, __m512& maxab);
  static void MinMax8(__m512i &a, __m512i &b);
  static void MinMax8(__m512d &a, __m512d &b);

  // BitonicSort(Sorting Networks)
  template <typename T>
  static void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                             T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  static void BitonicSort16x16(T &r0, T &r1, T &r2, T &r3,
                               T &r4, T &r5, T &r6, T &r7,
                               T &r8, T &r9, T &r10, T &r11,
                               T &r12, T &r13, T &r14, T &r15);

  // Transpose(Bitonic)
  static void Transpose8x8(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                           __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7);
  static void Transpose8x8(__m512d &row0, __m512d &row1, __m512d &row2, __m512d &row3,
                           __m512d &row4, __m512d &row5, __m512d &row6, __m512d &row7);
  static void Transpose16x16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                             __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
                             __m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11,
                             __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15);
  static void Transpose16x16(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                             __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7,
                             __m512 &row8, __m512 &row9, __m512 &row10, __m512 &row11,
                             __m512 &row12, __m512 &row13, __m512 &row14, __m512 &row15);

  static __m512i Reverse8(__m512i& v);
  static __m512i Reverse8(__m512d& v);
  static __m512i Reverse16(__m512i& v);
  static __m512i Reverse16(__m512& v);

  static void IntraRegisterSort8x8(__m512i& a8, __m512i& b8);
  static void IntraRegisterSort8x8(__m512d& a8, __m512d& b8);
  static void IntraRegisterSort16x16(__m512& a16, __m512& b16);
  static void IntraRegisterSort16x16(__m512i& a16, __m512i& b16);

  static void BitonicMerge8(__m512i& a, __m512i& b);
  static void BitonicMerge8(__m512d& a, __m512d& b);
  static void BitonicMerge16(__m512i& a, __m512i& b);
  static void BitonicMerge16(__m512& a, __m512& b);
};

#endif