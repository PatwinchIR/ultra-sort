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
  static void IntraRegisterSort8x8(__m256i& a8, __m256i& b8);
  static void IntraRegisterSort8x8(__m256& a8, __m256& b8);
  static void IntraRegisterSort4x4(__m256i& a4, __m256i& b4);
  static void IntraRegisterSort4x4(__m256d& a4, __m256d& b4);
  template <typename T>
  static void BitonicMerge8(T& a, T& b);
  template <typename T>
  static void BitonicMerge4(T& a, T& b);
};

#endif