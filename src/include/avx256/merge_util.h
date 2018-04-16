#pragma once

#include "avx256/merge_util.h"
#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

class MergeUtil{
 public:
  static void MergeRuns8(int *&arr, int N);
  static void MergePass8(int *&arr, int *buffer, int N, int run_size);
};

#endif