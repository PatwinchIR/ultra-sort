//
// Created by Dee Dong on 4/14/18.
//

#ifndef ULTRASORT_SIMD_SORT_AVX512_H
#define ULTRASORT_SIMD_SORT_AVX512_H

#endif //ULTRASORT_SIMD_SORT_AVX512_H

#pragma once

#include "common.h"
#include <cassert>
#include <cstdio>
#include <algorithm>

#define VECWIDTH_AVX2 512
void sort_block_avx512(int *&arr, int start, int network_size=8);
void merge_runs_avx512(int *&arr, int N, int network_size=8);
void merge_pass_avx512(int *&arr, int *buffer, int N, int run_size);
void bitonic_sort_avx512(__m512i& a, __m512i& b);
void sort_avx512(size_t N, int *&arr, int network_size=8);