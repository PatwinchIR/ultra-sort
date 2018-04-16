#pragma once

#include <iostream>
#include <x86intrin.h>
#include <string>
#include <cstdint>
#include <cassert>

/**
 * Common definitions
 */

#define SIMD_WIDTH 256

#if SIMD_WIDTH == 256
#if __AVX2__
#define AVX2
#else
#error "AVX2 not available on this platform"
#endif
#elif SIMD_WIDTH == 512
#if __AVX512F__
#define AVX512
#else
#error "AVX512 not available on this platform"
#endif
#endif

/**
 * Common utility functions
 */

/**
 * Templated function for memory aligned initialization
 * @tparam T: data type
 * @param ptr: reference to pointer needing init
 * @param N: size of data
 * @param alignment_size: alignment chunk size
 */
template <typename T>
void aligned_init(T* &ptr, int N, size_t alignment_size=64);

void print_arr(int *arr, int i, int j, const std::string &tag="");
void printkv_arr(int64_t *arr, int i, int j, const std::string &tag="");
