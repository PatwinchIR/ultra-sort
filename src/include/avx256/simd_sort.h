#pragma once

#include "avx256/simd_sort.h"
#include "avx256/sort_util.h"
#include "avx256/merge_util.h"
#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2
class SIMDSorter{
 public:
  static void SIMDSort32(size_t N, int *&arr);
  static void SIMDSort64(size_t N, int64_t *&arr);
};
#endif