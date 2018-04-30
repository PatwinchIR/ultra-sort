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
  // 32-bit
  static void MinMax8(__m256i &a, __m256i &b);
  static void MinMax8(const __m256i& a, const __m256i& b,
                      __m256i& minab, __m256i& maxab);
  static void MinMax8(__m256 &a, __m256 &b);
  static void MinMax8(const __m256& a, const __m256& b,
                      __m256& minab, __m256& maxab);
  // 64-bit
  static void MinMax4(__m256i &a, __m256i &b);
  static void MinMax4(__m256d &a, __m256d &b);

  // 32-bit Key-Value pairs
  static void MaskedMinMax8(__m256i &a, __m256i &b);
  static void MaskedMinMax8(__m256 &a, __m256 &b);
  // 64-bit Key-Value pairs
  static void MaskedMinMax4(__m256i &a, __m256i &b);
  static void MaskedMinMax4(__m256d &a, __m256d &b);

  // Transpose(Bitonic)
  static void Transpose8x8(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                           __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7);
  static void Transpose8x8(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                           __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7);
  template <typename T>
  static void Transpose4x4(T &row0, T &row1, T &row2, T &row3);

  static __m256i Reverse8(__m256i& v);
  static __m256 Reverse8(__m256& v);
  static __m256i Reverse4(__m256i& v);
  static __m256d Reverse4(__m256d& v);
  static __m256i MaskedReverse8(__m256i& v);
  static __m256 MaskedReverse8(__m256& v);
};

#endif