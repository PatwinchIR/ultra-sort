#include "avx256/sort_util.h"

#ifdef AVX2
namespace avx2 {
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

template<typename InType, typename RegType>
void MaskedSortBlock2x4(InType *&arr, int offset) {
  int ROW_SIZE = 4;
  // Put into registers
  RegType r0, r1;

  LoadReg(r0, arr + offset);
  LoadReg(r1, arr + offset + ROW_SIZE);

  // Apply bitonic sort
  MaskedBitonicSort2x4(r0, r1);

  // transpose(shuffle) to bring in order
  Transpose2x2(r0, r1);

  // restore into array
  StoreReg(r0, arr + offset);
  StoreReg(r1, arr + offset + ROW_SIZE);
}

template void MaskedSortBlock2x4<int64_t, __m256i>(int64_t *&arr, int offset);
template void MaskedSortBlock2x4<double, __m256d>(double *&arr, int offset);
}
#endif

