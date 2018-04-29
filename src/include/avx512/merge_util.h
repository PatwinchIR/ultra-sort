#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512MergeUtil{
 public:
  template <typename InType, typename RegType>
  static void MergeRuns16(InType *&arr, int N);
  template <typename InType, typename RegType>
  static void MergeRuns8(InType *&arr, int N);
  template <typename InType, typename RegType>
  static void MergePass16(InType *&arr, InType *buffer, int N, int run_size);
  template <typename InType, typename RegType>
  static void MergePass8(InType *&arr, InType *buffer, int N, int run_size);

  static void BitonicMerge8(__m512i& a, __m512i& b);
  static void BitonicMerge8(__m512d& a, __m512d& b);
  static void BitonicMerge16(__m512i& a, __m512i& b);
  static void BitonicMerge16(__m512& a, __m512& b);
};

#endif
