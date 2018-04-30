#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512MergeUtil{
 public:
  template <typename InType, typename RegType>
  void MergeRuns16(InType *&arr, int N);
  template <typename InType, typename RegType>
  void MergeRuns8(InType *&arr, int N);
  template <typename InType, typename RegType>
  void MergePass16(InType *&arr, InType *buffer, int N, int run_size);
  template <typename InType, typename RegType>
  void MergePass8(InType *&arr, InType *buffer, int N, int run_size);
};

#endif
