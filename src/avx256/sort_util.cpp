#include "avx256/sort_util.h"

#ifdef AVX2
template <typename InType, typename RegType>
void AVX256SortUtil::SortBlock64(InType *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7;

  AVX256Util::LoadReg(r0, arr + offset);
  AVX256Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::LoadReg(r3, arr + offset + ROW_SIZE*3);
  AVX256Util::LoadReg(r4, arr + offset + ROW_SIZE*4);
  AVX256Util::LoadReg(r5, arr + offset + ROW_SIZE*5);
  AVX256Util::LoadReg(r6, arr + offset + ROW_SIZE*6);
  AVX256Util::LoadReg(r7, arr + offset + ROW_SIZE*7);

  // Apply bitonic sort
  AVX256SortUtil::BitonicSort8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // transpose(shuffle) to bring in order
  AVX256Util::Transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // restore into array
  AVX256Util::StoreReg(r0, arr + offset);
  AVX256Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
  AVX256Util::StoreReg(r4, arr + offset + ROW_SIZE*4);
  AVX256Util::StoreReg(r5, arr + offset + ROW_SIZE*5);
  AVX256Util::StoreReg(r6, arr + offset + ROW_SIZE*6);
  AVX256Util::StoreReg(r7, arr + offset + ROW_SIZE*7);
}

template void AVX256SortUtil::SortBlock64<int, __m256i>(int *&arr, int offset);
template void AVX256SortUtil::SortBlock64<float, __m256>(float *&arr, int offset);

template <typename InType, typename RegType>
void AVX256SortUtil::MaskedSortBlock64(InType *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7;

  AVX256Util::LoadReg(r0, arr + offset);
  AVX256Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::LoadReg(r3, arr + offset + ROW_SIZE*3);
  AVX256Util::LoadReg(r4, arr + offset + ROW_SIZE*4);
  AVX256Util::LoadReg(r5, arr + offset + ROW_SIZE*5);
  AVX256Util::LoadReg(r6, arr + offset + ROW_SIZE*6);
  AVX256Util::LoadReg(r7, arr + offset + ROW_SIZE*7);

  // Apply bitonic sort
  AVX256SortUtil::MaskedBitonicSort8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // transpose(shuffle) to bring in order
  AVX256Util::Transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // restore into array
  AVX256Util::StoreReg(r0, arr + offset);
  AVX256Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
  AVX256Util::StoreReg(r4, arr + offset + ROW_SIZE*4);
  AVX256Util::StoreReg(r5, arr + offset + ROW_SIZE*5);
  AVX256Util::StoreReg(r6, arr + offset + ROW_SIZE*6);
  AVX256Util::StoreReg(r7, arr + offset + ROW_SIZE*7);
}

template void AVX256SortUtil::MaskedSortBlock64<int, __m256i>(int *&arr, int offset);
template void AVX256SortUtil::MaskedSortBlock64<float, __m256>(float *&arr, int offset);

template <typename InType, typename RegType>
void AVX256SortUtil::SortBlock16(InType *&arr, int offset) {
  int ROW_SIZE = 4;
  // Put into registers
  RegType r0, r1, r2, r3;
  AVX256Util::LoadReg(r0, arr + offset);
  AVX256Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::LoadReg(r3, arr + offset + ROW_SIZE*3);

  // Apply bitonic sort
  AVX256SortUtil::BitonicSort4x4(r0, r1, r2, r3);

  // transpose(shuffle) to bring in order
  AVX256Util::Transpose4x4(r0, r1, r2, r3);

  // restore into array
  AVX256Util::StoreReg(r0, &arr[offset]);
  AVX256Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
}

template void AVX256SortUtil::SortBlock16<int64_t, __m256i>(int64_t *&arr, int offset);
template void AVX256SortUtil::SortBlock16<double, __m256d>(double *&arr, int offset);

/**
 * Bitonic Sorting networks:
 * 8x8 networks: int32, float32
 * 4x4 networks: int64
 */
template <typename T>
void AVX256SortUtil::BitonicSort8x8(T &r0,
                                T &r1,
                                T &r2,
                                T &r3,
                                T &r4,
                                T &r5,
                                T &r6,
                                T &r7) {
  AVX256Util::MinMax8(r0, r1);
  AVX256Util::MinMax8(r2, r3);
  AVX256Util::MinMax8(r4, r5);
  AVX256Util::MinMax8(r6, r7);
  AVX256Util::MinMax8(r0, r2);
  AVX256Util::MinMax8(r4, r6);
  AVX256Util::MinMax8(r1, r3);
  AVX256Util::MinMax8(r5, r7);
  AVX256Util::MinMax8(r1, r2);
  AVX256Util::MinMax8(r5, r6);
  AVX256Util::MinMax8(r0, r4);
  AVX256Util::MinMax8(r1, r5);
  AVX256Util::MinMax8(r1, r4);
  AVX256Util::MinMax8(r2, r6);
  AVX256Util::MinMax8(r3, r7);
  AVX256Util::MinMax8(r3, r6);
  AVX256Util::MinMax8(r2, r4);
  AVX256Util::MinMax8(r3, r5);
  AVX256Util::MinMax8(r3, r4);
}

// 32 bit ints, floats
template void AVX256SortUtil::BitonicSort8x8<__m256i>(__m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&);
template void AVX256SortUtil::BitonicSort8x8<__m256>(__m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&);

template <typename T>
void AVX256SortUtil::MaskedBitonicSort8x8(T &r0,
                                          T &r1,
                                          T &r2,
                                          T &r3,
                                          T &r4,
                                          T &r5,
                                          T &r6,
                                          T &r7) {
  AVX256Util::MaskedMinMax8(r0, r1);
  AVX256Util::MaskedMinMax8(r2, r3);
  AVX256Util::MaskedMinMax8(r4, r5);
  AVX256Util::MaskedMinMax8(r6, r7);
  AVX256Util::MaskedMinMax8(r0, r2);
  AVX256Util::MaskedMinMax8(r4, r6);
  AVX256Util::MaskedMinMax8(r1, r3);
  AVX256Util::MaskedMinMax8(r5, r7);
  AVX256Util::MaskedMinMax8(r1, r2);
  AVX256Util::MaskedMinMax8(r5, r6);
  AVX256Util::MaskedMinMax8(r0, r4);
  AVX256Util::MaskedMinMax8(r1, r5);
  AVX256Util::MaskedMinMax8(r1, r4);
  AVX256Util::MaskedMinMax8(r2, r6);
  AVX256Util::MaskedMinMax8(r3, r7);
  AVX256Util::MaskedMinMax8(r3, r6);
  AVX256Util::MaskedMinMax8(r2, r4);
  AVX256Util::MaskedMinMax8(r3, r5);
  AVX256Util::MaskedMinMax8(r3, r4);
}

// 32 bit KV ints, floats
template void AVX256SortUtil::MaskedBitonicSort8x8<__m256i>(__m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&);
template void AVX256SortUtil::MaskedBitonicSort8x8<__m256>(__m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&);


template <typename T>
void AVX256SortUtil::BitonicSort4x4(T &r0,
                                T &r1,
                                T &r2,
                                T &r3) {
  AVX256Util::MinMax4(r0, r1);
  AVX256Util::MinMax4(r2, r3);
  AVX256Util::MinMax4(r0, r2);
  AVX256Util::MinMax4(r1, r3);
  AVX256Util::MinMax4(r1, r2);
}

// 64 bit ints, floats
template void AVX256SortUtil::BitonicSort4x4<__m256i>(__m256i&, __m256i&, __m256i&, __m256i&);
template void AVX256SortUtil::BitonicSort4x4<__m256d>(__m256d&, __m256d&, __m256d&, __m256d&);

#endif

