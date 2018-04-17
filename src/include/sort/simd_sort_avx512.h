#pragma once

#include "common.h"
#include <cassert>
#include <cstdio>
#include <algorithm>

#ifdef AVX512
#ifdef __AVX512F__

#define VECWIDTH_AVX2 512
void sort_block_avx512(int *&arr, int start, int network_size=8);
void merge_runs_avx512(int *&arr, int N, int network_size=8);
void merge_pass_avx512(int *&arr, int *buffer, int N, int run_size);
void bitonic_sort_avx512(__m512i& a, __m512i& b);
void sort_avx512(size_t N, int *&arr, int network_size=8);

#endif
#endif