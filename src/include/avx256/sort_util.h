#pragma once

#include "avx256/sort_util.h"
#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

class SortUtil{
 public:
  static void SortBlock64(int *&arr, int offset);
};

#endif