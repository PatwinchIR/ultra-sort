#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

namespace avx512{
  template <typename InType, typename RegType>
  void SortBlock256(InType *&arr, int offset);
  template <typename InType, typename RegType>
  void SortBlock64(InType *&arr, int offset);

  // Masked
  template <typename InType, typename RegType>
  void MaskedSortBlock8x16(InType *&arr, size_t offset);
  template <typename InType, typename RegType>
  void MaskedSortBlock4x8(InType *&arr, size_t offset);

};

#endif
