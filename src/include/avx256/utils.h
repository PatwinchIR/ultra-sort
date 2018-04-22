#pragma once

#include "common.h"

#ifdef AVX2

class AVX256Util{
 public:
  // Load/Stores
  template <typename InType, typename RegType>
  static void LoadReg(RegType &r, InType* arr);
  template <typename InType, typename RegType>
  static void StoreReg(const RegType &r, InType* arr);

  // Converters
  static __m256d Int64ToDoubleReg(const __m256i &repi64);
  static __m256i DoubleToInt64Reg(const __m256d &rd);

  // Min/Max
  static void MinMax8(__m256i &a, __m256i &b);
  static void MinMax8(const __m256i& a, const __m256i& b,
                      __m256i& minab, __m256i& maxab);
  static void MinMax8(__m256 &a, __m256 &b);
  static void MinMax8(const __m256& a, const __m256& b,
                      __m256& minab, __m256& maxab);
  static void MinMax4(__m256i &a, __m256i &b);
  static void MinMax4(__m256d &a, __m256d &b);

  // BitonicSort(Sorting Networks)
  template <typename T>
  static void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                             T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  static void BitonicSort4x4(T &r0, T &r1, T &r2, T &r3);
  // Transpose(Bitonic)
  static void Transpose8x8(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                           __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7);
  static void Transpose8x8(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                           __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7);
  static void Transpose4x4(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3);
  static void Transpose4x4(__m256d &row0, __m256d &row1, __m256d &row2, __m256d &row3);
  static __m256i Reverse8(__m256i& v);
  static __m256 Reverse8(__m256& v);
  static __m256i Reverse4(__m256i& v);
  static __m256d Reverse4(__m256d& v);
  static void IntraRegisterSort8x8(__m256i& a8, __m256i& b8);
  static void IntraRegisterSort8x8(__m256& a8, __m256& b8);
  static void IntraRegisterSort4x4(__m256i& a4, __m256i& b4);
  static void IntraRegisterSort4x4(__m256d& a4, __m256d& b4);
  template <typename T>
  static void BitonicMerge8(T& a, T& b);
  template <typename T>
  static void BitonicMerge4(T& a, T& b);
};

#endif