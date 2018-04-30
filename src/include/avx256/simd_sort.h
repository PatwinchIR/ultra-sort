#pragma once

#include "avx256/sort_util.h"
#include "avx256/merge_util.h"
#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2
namespace avx2{
  void SIMDSort(size_t N, int *&arr);
  void SIMDSort(size_t N, int64_t *&arr);
  void SIMDSort(size_t N, float *&arr);
  void SIMDSort(size_t N, double *&arr);
  void SIMDSort(size_t N, std::pair<int,int> *&arr);
};
#endif