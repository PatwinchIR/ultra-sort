#pragma once

#include "avx256/sort_util.h"
#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

class SortUtil{
 public:
  template <typename InType, typename RegType>
  static void SortBlock64(InType *&arr, int offset);
  template <typename InType, typename RegType>
  static void SortBlock16(InType *&arr, int offset);
};

#endif