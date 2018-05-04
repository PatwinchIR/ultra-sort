#include "avx512/sort_util.h"

#ifdef AVX512
namespace avx512 {
template<typename InType, typename RegType>
void SortBlock256(InType *&arr, size_t offset) {
  int ROW_SIZE = 16;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  LoadReg(r0, arr + offset);
  LoadReg(r1, arr + offset + ROW_SIZE);
  LoadReg(r2, arr + offset + ROW_SIZE * 2);
  LoadReg(r3, arr + offset + ROW_SIZE * 3);
  LoadReg(r4, arr + offset + ROW_SIZE * 4);
  LoadReg(r5, arr + offset + ROW_SIZE * 5);
  LoadReg(r6, arr + offset + ROW_SIZE * 6);
  LoadReg(r7, arr + offset + ROW_SIZE * 7);
  LoadReg(r8, arr + offset + ROW_SIZE * 8);
  LoadReg(r9, arr + offset + ROW_SIZE * 9);
  LoadReg(r10, arr + offset + ROW_SIZE * 10);
  LoadReg(r11, arr + offset + ROW_SIZE * 11);
  LoadReg(r12, arr + offset + ROW_SIZE * 12);
  LoadReg(r13, arr + offset + ROW_SIZE * 13);
  LoadReg(r14, arr + offset + ROW_SIZE * 14);
  LoadReg(r15, arr + offset + ROW_SIZE * 15);

  // Apply bitonic sort
  BitonicSort16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

  // transpose(shuffle) to bring in order
  Transpose16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

  // restore into array
  StoreReg(r0, arr + offset);
  StoreReg(r1, arr + offset + ROW_SIZE);
  StoreReg(r2, arr + offset + ROW_SIZE * 2);
  StoreReg(r3, arr + offset + ROW_SIZE * 3);
  StoreReg(r4, arr + offset + ROW_SIZE * 4);
  StoreReg(r5, arr + offset + ROW_SIZE * 5);
  StoreReg(r6, arr + offset + ROW_SIZE * 6);
  StoreReg(r7, arr + offset + ROW_SIZE * 7);
  StoreReg(r8, arr + offset + ROW_SIZE * 8);
  StoreReg(r9, arr + offset + ROW_SIZE * 9);
  StoreReg(r10, arr + offset + ROW_SIZE * 10);
  StoreReg(r11, arr + offset + ROW_SIZE * 11);
  StoreReg(r12, arr + offset + ROW_SIZE * 12);
  StoreReg(r13, arr + offset + ROW_SIZE * 13);
  StoreReg(r14, arr + offset + ROW_SIZE * 14);
  StoreReg(r15, arr + offset + ROW_SIZE * 15);
}

template void SortBlock256<int, __m512i>(int *&arr, size_t offset);
template void SortBlock256<float, __m512>(float *&arr, size_t offset);

template<typename InType, typename RegType>
void MaskedSortBlock8x16(InType *&arr, size_t offset) {
  int ROW_SIZE = 16;
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
  MaskedBitonicSort8x16(r0, r1, r2, r3, r4, r5, r6, r7);

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

template void MaskedSortBlock8x16<int, __m512i>(int *&arr, size_t offset);
template void MaskedSortBlock8x16<float, __m512>(float *&arr, size_t offset);

template<typename InType, typename RegType>
void SortBlock64(InType *&arr, size_t offset) {
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
  StoreReg(r0, &arr[offset]);
  StoreReg(r1, arr + offset + ROW_SIZE);
  StoreReg(r2, arr + offset + ROW_SIZE * 2);
  StoreReg(r3, arr + offset + ROW_SIZE * 3);
  StoreReg(r4, arr + offset + ROW_SIZE * 4);
  StoreReg(r5, arr + offset + ROW_SIZE * 5);
  StoreReg(r6, arr + offset + ROW_SIZE * 6);
  StoreReg(r7, arr + offset + ROW_SIZE * 7);
}

template void SortBlock64<int64_t, __m512i>(int64_t *&arr, size_t offset);
template void SortBlock64<double, __m512d>(double *&arr, size_t offset);

template<typename InType, typename RegType>
void MaskedSortBlock4x8(InType *&arr, size_t offset) {
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

template void MaskedSortBlock4x8<int64_t, __m512i>(int64_t *&arr, size_t offset);
template void MaskedSortBlock4x8<double, __m512d>(double *&arr, size_t offset);

}

#endif

