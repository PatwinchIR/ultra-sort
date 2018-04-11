#pragma once

#include <iostream>
#include <x86intrin.h>

/**
 * Common definitions
 */

#define NETWORK_SIZE 8

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
void aligned_init(T* &ptr, int N, size_t alignment_size=64) {
  if (posix_memalign((void **)&ptr, alignment_size, N*sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
}