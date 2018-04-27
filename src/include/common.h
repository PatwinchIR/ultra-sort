#pragma once

#include <iostream>
#include <x86intrin.h>
#include <string>
#include <cstdint>
#include <cassert>

/**
 * Common definitions
 */


#if __AVX2__
#define AVX2
#endif

#if __AVX512F__
#define AVX512
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

template <typename T>
void print_arr(T *arr, int i, int j, const std::string &tag="");

void print_kvarr(std::pair<int,int> *arr, int i, int j, const std::string &tag="");
void print_kvarr(int64_t *arr, int i, int j, const std::string &tag="");
