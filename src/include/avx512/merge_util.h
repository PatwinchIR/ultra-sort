#pragma once

#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

namespace avx512{
  template <typename InType, typename RegType>
  void MergeRuns16(InType *&arr, size_t N);
  template<typename InType, typename RegType>
  void MaskedMergeRuns16(InType *&arr, size_t N);
  template <typename InType, typename RegType>
  void MergeRuns8(InType *&arr, size_t N);
  template<typename InType, typename RegType>
  void MaskedMergeRuns8(InType *&arr, size_t N);
  template <typename InType, typename RegType>
  void MergePass16(InType *&arr, InType *buffer, size_t N, unsigned int run_size);
  template <typename InType, typename RegType>
  void MaskedMergePass16(InType *&arr, InType *buffer, size_t N, unsigned int run_size);
  template <typename InType, typename RegType>
  void MergePass8(InType *&arr, InType *buffer, size_t N, unsigned int run_size);
  template <typename InType, typename RegType>
  void MaskedMergePass8(InType *&arr, InType *buffer, size_t N, unsigned int run_size);
};

#endif
