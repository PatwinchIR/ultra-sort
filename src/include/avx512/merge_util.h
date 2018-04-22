#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

class AVX512MergeUtil{
 public:
  static void MergeRuns16(int *&arr, int N);
  static void MergeRuns8(int64_t *&arr, int N);
  static void MergePass16(int *&arr, int *buffer, int N, int run_size);
  static void MergePass8(int64_t *&arr, int64_t *buffer, int N, int run_size);
};

#endif
