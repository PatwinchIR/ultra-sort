#pragma once

#include "common.h"

#ifdef AVX2

namespace avx2{
// Relevant consts
const __m256i KEYCOPY_FLAG_32 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6);
const __m256i REVERSE_FLAG_32 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
const __m256i MASK_REVERSE_FLAG_32 = _mm256_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1);
const __m256i FLIP_HALVES_FLAG = _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4);

// Load/Stores
template <typename InType, typename RegType>
void LoadReg(RegType &r, InType* arr);
template <typename InType, typename RegType>
void StoreReg(const RegType &r, InType* arr);

// Converters
__m256d Int64ToDoubleReg(const __m256i &repi64);
__m256i DoubleToInt64Reg(const __m256d &rd);

// Min/Max
// 32-bit
void MinMax8(__m256i &a, __m256i &b);
void MinMax8(const __m256i& a, const __m256i& b,
             __m256i& minab, __m256i& maxab);
void MinMax8(__m256 &a, __m256 &b);
void MinMax8(const __m256& a, const __m256& b,
             __m256& minab, __m256& maxab);
// 64-bit
void MinMax4(__m256i &a, __m256i &b);
void MinMax4(__m256d &a, __m256d &b);

// 32-bit Key-Value pairs
void MaskedMinMax8(__m256i &a, __m256i &b);
void MaskedMinMax8(__m256 &a, __m256 &b);
// 64-bit Key-Value pairs
void MaskedMinMax4(__m256i &a, __m256i &b);
void MaskedMinMax4(__m256d &a, __m256d &b);

// BitonicSort(Sorting Networks)
// Simple
template <typename T>
void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                    T &r4, T &r5, T &r6, T &r7);
template <typename T>
void BitonicSort4x4(T &r0, T &r1, T &r2, T &r3);
// Masked
template <typename T>
void MaskedBitonicSort4x8(T &r0, T &r1, T &r2, T &r3);
template <typename T>
void MaskedBitonicSort2x4(T &r0, T &r1);

// Transpose(Bitonic)
void Transpose8x8(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3,
                  __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7);
void Transpose8x8(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                  __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7);
template <typename T>
void Transpose4x4(T &row0, T &row1, T &row2, T &row3);

void Transpose2x2(__m256d &row0, __m256d &row1);
void Transpose2x2(__m256i &row0, __m256i &row1);

template <typename T>
void Reverse8(T& v);
template <typename T>
void Reverse4(T& v);

template <typename T>
void MaskedReverse8(T& v);
template <typename T>
void MaskedReverse4(T& v);

/**
 * Note: Templatizing IntraRegSorts incurs significant overhead
 * (and may not be compatible with certain compilers)
 * https://stackoverflow.com/questions/50109218/casting-with-avx-intrinsics
 * There are many internal casts to have this working correctly.
 */

// since there are many typ
// Simple IntraReg Sorts
void IntraRegisterSort8x8(__m256i& a8, __m256i& b8);
void IntraRegisterSort8x8(__m256& a8, __m256& b8);
template <typename T>
void IntraRegisterSort4x4(T& a4, T& b4);
// Masked IntraReg Sorts
template <typename T>
void MaskedIntraRegisterSort8x8(T& a4kv, T& b4kv);
template <typename T>
void BitonicMerge8(T& a, T& b);
template <typename T>
void BitonicMerge4(T& a, T& b);
template <typename T>
void MaskedBitonicMerge8(T& a, T& b);
// To simple for an IntraReg sort
template <typename T>
void MaskedBitonicMerge4(T& a, T& b);
};

#endif