#pragma once

#include "avx512/sort_util.h"
#include "avx512/merge_util.h"
#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512
class AVX512SIMDSorter{
 public:
  static void SIMDSort(size_t N, int *&arr);
  static void SIMDSort(size_t N, int64_t *&arr);
  static void SIMDSort(size_t N, float *&arr);
  static void SIMDSort(size_t N, double *&arr);
  static void SIMDSort32KV(size_t N, std::pair<int,int> *&arr);
};
#endif
