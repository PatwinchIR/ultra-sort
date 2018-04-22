#pragma once

#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

class AVX256MergeUtil{
 public:
  template <typename InType, typename RegType>
  static void MergeRuns8(InType *&arr, int N);
  template <typename InType, typename RegType>
  static void MergeRuns4(InType *&arr, int N);
  template <typename InType, typename RegType>
  static void MergePass8(InType *&arr, InType *buffer, int N, int run_size);
  template <typename InType, typename RegType>
  static void MergePass4(InType *&arr, InType *buffer, int N, int run_size);
};

#endif