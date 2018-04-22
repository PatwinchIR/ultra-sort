#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512SortUtil{
 public:
  static void SortBlock256(int *&arr, int offset);
  static void SortBlock64(int64_t *&arr, int offset);
};

#endif
