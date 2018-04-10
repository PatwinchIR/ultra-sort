#pragma once

#include <x86intrin.h>

#define NUMBITS(x) (sizeof(x) * 8)

#ifdef __AVX__
#define VECWIDTH_AVX 128
#endif

#ifdef __AVX2__
#define VECWIDTH_AVX2 256
void sort_block_avx2(int *arr, int start, int network_size=8);
void sort_avx2(size_t N, int *arr, int network_size=8);
#endif

#ifdef __AVX512F__
#define VECWIDTH_AVX512 512
#endif
