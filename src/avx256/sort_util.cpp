#include "avx256/sort_util.h"

#ifdef AVX2
namespace avx2 {
/**
 * Bitonic Sorting networks:
 * 8x8 networks: int32, float32
 * 4x4 networks: int64
 */
template<typename T>
void BitonicSort8x8(T &r0,
                    T &r1,
                    T &r2,
                    T &r3,
                    T &r4,
                    T &r5,
                    T &r6,
                    T &r7) {
  MinMax8(r0, r1);
  MinMax8(r2, r3);
  MinMax8(r4, r5);
  MinMax8(r6, r7);
  MinMax8(r0, r2);
  MinMax8(r4, r6);
  MinMax8(r1, r3);
  MinMax8(r5, r7);
  MinMax8(r1, r2);
  MinMax8(r5, r6);
  MinMax8(r0, r4);
  MinMax8(r1, r5);
  MinMax8(r1, r4);
  MinMax8(r2, r6);
  MinMax8(r3, r7);
  MinMax8(r3, r6);
  MinMax8(r2, r4);
  MinMax8(r3, r5);
  MinMax8(r3, r4);
}

// 32 bit ints, floats
template void BitonicSort8x8<__m256i>(__m256i &,
                                      __m256i &,
                                      __m256i &,
                                      __m256i &,
                                      __m256i &,
                                      __m256i &,
                                      __m256i &,
                                      __m256i &);
template void BitonicSort8x8<__m256>(__m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &, __m256 &);

template<typename T>
void BitonicSort4x4(T &r0,
                    T &r1,
                    T &r2,
                    T &r3) {
  MinMax4(r0, r1);
  MinMax4(r2, r3);
  MinMax4(r0, r2);
  MinMax4(r1, r3);
  MinMax4(r1, r2);
}

// 64 bit ints, floats
template void BitonicSort4x4<__m256i>(__m256i &, __m256i &, __m256i &, __m256i &);
template void BitonicSort4x4<__m256d>(__m256d &, __m256d &, __m256d &, __m256d &);

template<typename T>
void MaskedBitonicSort4x8(T &r0,
                          T &r1,
                          T &r2,
                          T &r3) {
  MaskedMinMax8(r0, r1);
  MaskedMinMax8(r2, r3);
  MaskedMinMax8(r0, r2);
  MaskedMinMax8(r1, r3);
  MaskedMinMax8(r1, r2);
}

// 32 bit KV ints, floats
template void MaskedBitonicSort4x8<__m256i>(__m256i &, __m256i &, __m256i &, __m256i &);
template void MaskedBitonicSort4x8<__m256>(__m256 &, __m256 &, __m256 &, __m256 &);

template<typename InType, typename RegType>
void SortBlock64(InType *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7;

  LoadReg(r0, arr + offset);
  LoadReg(r1, arr + offset + ROW_SIZE);
  LoadReg(r2, arr + offset + ROW_SIZE * 2);
  LoadReg(r3, arr + offset + ROW_SIZE * 3);
  LoadReg(r4, arr + offset + ROW_SIZE * 4);
  LoadReg(r5, arr + offset + ROW_SIZE * 5);
  LoadReg(r6, arr + offset + ROW_SIZE * 6);
  LoadReg(r7, arr + offset + ROW_SIZE * 7);

  // Apply bitonic sort
  BitonicSort8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // transpose(shuffle) to bring in order
  Transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // restore into array
  StoreReg(r0, arr + offset);
  StoreReg(r1, arr + offset + ROW_SIZE);
  StoreReg(r2, arr + offset + ROW_SIZE * 2);
  StoreReg(r3, arr + offset + ROW_SIZE * 3);
  StoreReg(r4, arr + offset + ROW_SIZE * 4);
  StoreReg(r5, arr + offset + ROW_SIZE * 5);
  StoreReg(r6, arr + offset + ROW_SIZE * 6);
  StoreReg(r7, arr + offset + ROW_SIZE * 7);
}

template void SortBlock64<int, __m256i>(int *&arr, int offset);
template void SortBlock64<float, __m256>(float *&arr, int offset);

template<typename InType, typename RegType>
void MaskedSortBlock4x8(InType *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  RegType r0, r1, r2, r3;

  LoadReg(r0, arr + offset);
  LoadReg(r1, arr + offset + ROW_SIZE);
  LoadReg(r2, arr + offset + ROW_SIZE * 2);
  LoadReg(r3, arr + offset + ROW_SIZE * 3);

  // Apply bitonic sort
  MaskedBitonicSort4x8(r0, r1, r2, r3);

  // transpose(shuffle) to bring in order
  Transpose4x4(r0, r1, r2, r3);

  // restore into array
  StoreReg(r0, arr + offset);
  StoreReg(r1, arr + offset + ROW_SIZE);
  StoreReg(r2, arr + offset + ROW_SIZE * 2);
  StoreReg(r3, arr + offset + ROW_SIZE * 3);
}

template void MaskedSortBlock4x8<int, __m256i>(int *&arr, int offset);
template void MaskedSortBlock4x8<float, __m256>(float *&arr, int offset);

template<typename InType, typename RegType>
void SortBlock16(InType *&arr, int offset) {
  int ROW_SIZE = 4;
  // Put into registers
  RegType r0, r1, r2, r3;
  LoadReg(r0, arr + offset);
  LoadReg(r1, arr + offset + ROW_SIZE);
  LoadReg(r2, arr + offset + ROW_SIZE * 2);
  LoadReg(r3, arr + offset + ROW_SIZE * 3);

  // Apply bitonic sort
  BitonicSort4x4(r0, r1, r2, r3);

  // transpose(shuffle) to bring in order
  Transpose4x4(r0, r1, r2, r3);

  // restore into array
  StoreReg(r0, &arr[offset]);
  StoreReg(r1, arr + offset + ROW_SIZE);
  StoreReg(r2, arr + offset + ROW_SIZE * 2);
  StoreReg(r3, arr + offset + ROW_SIZE * 3);
}

template void SortBlock16<int64_t, __m256i>(int64_t *&arr, int offset);
template void SortBlock16<double, __m256d>(double *&arr, int offset);
}
#endif

