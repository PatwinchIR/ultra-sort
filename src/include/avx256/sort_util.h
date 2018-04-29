#pragma once

#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

class AVX256SortUtil{
 public:
  template <typename InType, typename RegType>
  static void SortBlock64(InType *&arr, int offset);
  template <typename InType, typename RegType>
  static void MaskedSortBlock64(InType *&arr, int offset);
  template <typename InType, typename RegType>
  static void SortBlock16(InType *&arr, int offset);

  // BitonicSort(Sorting Networks)
  // Simple
  template <typename T>
  static void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                             T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  static void BitonicSort4x4(T &r0, T &r1, T &r2, T &r3);
  // Masked
  template <typename T>
  static void MaskedBitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                                   T &r4, T &r5, T &r6, T &r7);
};

#endif