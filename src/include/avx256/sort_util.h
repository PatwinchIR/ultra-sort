#pragma once

#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

namespace avx2{
  // Regular
  template <typename InType, typename RegType>
  void SortBlock64(InType *&arr, int offset);
  template <typename InType, typename RegType>
  void SortBlock16(InType *&arr, int offset);

  // Masked
  template <typename InType, typename RegType>
  void MaskedSortBlock4x8(InType *&arr, int offset);
  template <typename InType, typename RegType>
  void MaskedSortBlock2x4(InType *&arr, int offset);

};

#endif