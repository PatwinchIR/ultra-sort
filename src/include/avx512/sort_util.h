#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512SortUtil{
 public:
  template <typename InType, typename RegType>
  void SortBlock256(InType *&arr, int offset);
  template <typename InType, typename RegType>
<<<<<<< HEAD
  void SortBlock64(InType *&arr, int offset);
=======
  static void SortBlock64(InType *&arr, int offset);

  // BitonicSort(Sorting Networks)
  template <typename T>
  static void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                             T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  static void BitonicSort16x16(T &r0, T &r1, T &r2, T &r3,
                               T &r4, T &r5, T &r6, T &r7,
                               T &r8, T &r9, T &r10, T &r11,
                               T &r12, T &r13, T &r14, T &r15);
>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
};

#endif
