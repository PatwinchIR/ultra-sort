#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512SortUtil{
 public:
  template <typename InType, typename RegType>
  void SortBlock256(InType *&arr, int offset);
  template <typename InType, typename RegType>
  void SortBlock64(InType *&arr, int offset);
};

#endif
