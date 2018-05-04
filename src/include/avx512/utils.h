#pragma once

#include "common.h"

#ifdef AVX512

namespace avx512{
  // Relevant consts
  const __m512i REVERSE_FLAG_32 = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i REVERSE_FLAG_64 = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
  const __m512i MASK_REVERSE_FLAG_32 = _mm512_setr_epi32(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
  const __m512i MASK_REVERSE_FLAG_64 = _mm512_setr_epi64(6, 7, 4, 5, 2, 3, 0, 1);

  const __m512i EXCHANGE_EACH = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6);

  const __m512i EXCHANGE_QUARTER_8 = _mm512_setr_epi64(2, 3, 0, 1, 6, 7, 4, 5);

  const __m512i EXCHANGE_HALF_8 = _mm512_setr_epi64(3, 2, 1, 0, 7, 6, 5, 4);

  const __m512i EXCHANGE_HALF_16 = _mm512_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
  const __m512i MASK_EXCHANGE_HALF_16 = _mm512_setr_epi32(6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10, 11, 8, 9);

  const __m512i BLEND_LO_128 = _mm512_setr_epi64(0, 1, 8, 9, 2, 3, 10, 11);
  const __m512i BLEND_HI_128 = _mm512_setr_epi64(4, 5, 12, 13, 6, 7, 14, 15);

  const __m512i BLEND_HALF_LO_128 = _mm512_setr_epi64(0, 1, 8, 9, 4, 5, 12, 13);
  const __m512i BLEND_HALF_HI_128 = _mm512_setr_epi64(2, 3, 10, 11, 6, 7, 14, 15);

  const __m512i BLEND_LO_256 = _mm512_setr_epi64(0, 1, 2, 3, 8, 9, 10, 11);
  const __m512i BLEND_HI_256 = _mm512_setr_epi64(4, 5, 6, 7, 12, 13, 14, 15);
  

  // Load/Stores
  template <typename InType, typename RegType>
  void LoadReg(RegType &r, InType* arr);
  template <typename InType, typename RegType>
  void StoreReg(const RegType &r, InType* arr);

//  // Converters
//  __m512d Int64ToDoubleReg(const __m512i &repi64);
//  __m512i DoubleToInt64Reg(const __m512d &rd);

  // Min/Max
  void MinMax16(__m512i &a, __m512i &b);
  void MinMax16(const __m512i& a, const __m512i& b,
                       __m512i& minab, __m512i& maxab);
  void MinMax16(__m512 &a, __m512 &b);
  void MinMax16(const __m512& a, const __m512& b,
                       __m512& minab, __m512& maxab);
  void MinMax8(__m512i &a, __m512i &b);
  void MinMax8(const __m512i &a, const __m512i &b,
                         __m512i& minab, __m512i& maxab);
  void MinMax8(__m512d &a, __m512d &b);
  void MinMax8(const __m512d &a, const __m512d &b,
                      __m512d& minab, __m512d& maxab);

  // 32-bit Key-Value pairs
  void MaskedMinMax16(__m512i &a, __m512i &b);
  void MaskedMinMax16(const __m512i& a, const __m512i& b,
                      __m512i& minab, __m512i& maxab);
  void MaskedMinMax16(__m512 &a, __m512 &b);
  void MaskedMinMax16(const __m512& a, const __m512& b,
                      __m512& minab, __m512& maxab);
  // 64-bit Key-Value pairs
  void MaskedMinMax8(__m512i &a, __m512i &b);
  void MaskedMinMax8(const __m512i &a, const __m512i &b,
               __m512i& minab, __m512i& maxab);
  void MaskedMinMax8(__m512d &a, __m512d &b);
  void MaskedMinMax8(const __m512d &a, const __m512d &b,
               __m512d& minab, __m512d& maxab);

  // BitonicSort(Sorting Networks)
  template <typename T>
  void BitonicSort8x8(T &r0, T &r1, T &r2, T &r3,
                      T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  void BitonicSort16x16(T &r0, T &r1, T &r2, T &r3,
                        T &r4, T &r5, T &r6, T &r7,
                        T &r8, T &r9, T &r10, T &r11,
                        T &r12, T &r13, T &r14, T &r15);

  // Masked
  template <typename T>
  void MaskedBitonicSort8x16(T &r0, T &r1, T &r2, T &r3,
                             T &r4, T &r5, T &r6, T &r7);
  template <typename T>
  void MaskedBitonicSort4x8(T &r0, T &r1, T &r2, T &r3);

  void Transpose4x4(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3);
  void Transpose4x4(__m512d &row0, __m512d &row1, __m512d &row2, __m512d &row3);

  // Transpose(Bitonic)
  void Transpose8x8(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                    __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7);
  void Transpose8x8(__m512d &row0, __m512d &row1, __m512d &row2, __m512d &row3,
                    __m512d &row4, __m512d &row5, __m512d &row6, __m512d &row7);
  void Transpose8x8(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                    __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7);
  void Transpose16x16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                      __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
                      __m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11,
                      __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15);
  void Transpose16x16(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                      __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7,
                      __m512 &row8, __m512 &row9, __m512 &row10, __m512 &row11,
                      __m512 &row12, __m512 &row13, __m512 &row14, __m512 &row15);


  void Reverse8(__m512i& v);
  void Reverse8(__m512d& v);
  void Reverse16(__m512i& v);
  void Reverse16(__m512& v);

  void MaskedReverse8(__m512i& v);
  void MaskedReverse8(__m512d& v);
  void MaskedReverse16(__m512i& v);
  void MaskedReverse16(__m512& v);

  template <typename T>
  void MaskedReverse8(T& v);
  template <typename T>
  void MaskedReverse16(T& v);

  void IntraRegisterSort8x8(__m512i& a8, __m512i& b8);
  void IntraRegisterSort8x8(__m512d& a8, __m512d& b8);
  void IntraRegisterSort16x16(__m512& a16, __m512& b16);
  void IntraRegisterSort16x16(__m512i& a16, __m512i& b16);

  void MaskedIntraRegisterSort8x8(__m512i &a8, __m512i &b8);
  void MaskedIntraRegisterSort8x8(__m512d &a8, __m512d &b8);
  void MaskedIntraRegisterSort16x16(__m512i &a16, __m512i &b16);
  void MaskedIntraRegisterSort16x16(__m512 &a16, __m512 &b16);

  void BitonicMerge8(__m512i& a, __m512i& b);
  void BitonicMerge8(__m512d& a, __m512d& b);
  void BitonicMerge16(__m512i& a, __m512i& b);
  void BitonicMerge16(__m512& a, __m512& b);

  template <typename T>
  void MaskedBitonicMerge16(T& a, T& b);
  // To simple for an IntraReg sort
  template <typename T>
  void MaskedBitonicMerge8(T& a, T& b);
};

#endif