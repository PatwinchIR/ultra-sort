#include "avx512/sort_util.h"

#ifdef AVX512
template <typename InType, typename RegType>
void AVX512SortUtil::SortBlock256(InType *&arr, int offset) {
  int ROW_SIZE = 16;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
  AVX512Util::LoadReg(r0, arr + offset);
  AVX512Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX512Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX512Util::LoadReg(r3, arr + offset + ROW_SIZE*3);
  AVX512Util::LoadReg(r4, arr + offset + ROW_SIZE*4);
  AVX512Util::LoadReg(r5, arr + offset + ROW_SIZE*5);
  AVX512Util::LoadReg(r6, arr + offset + ROW_SIZE*6);
  AVX512Util::LoadReg(r7, arr + offset + ROW_SIZE*7);
  AVX512Util::LoadReg(r8, arr + offset + ROW_SIZE*8);
  AVX512Util::LoadReg(r9, arr + offset + ROW_SIZE*9);
  AVX512Util::LoadReg(r10, arr + offset + ROW_SIZE*10);
  AVX512Util::LoadReg(r11, arr + offset + ROW_SIZE*11);
  AVX512Util::LoadReg(r12, arr + offset + ROW_SIZE*12);
  AVX512Util::LoadReg(r13, arr + offset + ROW_SIZE*13);
  AVX512Util::LoadReg(r14, arr + offset + ROW_SIZE*14);
  AVX512Util::LoadReg(r15, arr + offset + ROW_SIZE*15);

  // Apply bitonic sort
  AVX512Util::BitonicSort16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

  // transpose(shuffle) to bring in order
  AVX512Util::Transpose16x16(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

  // restore into array
  AVX512Util::StoreReg(r0, arr + offset);
  AVX512Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX512Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX512Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
  AVX512Util::StoreReg(r4, arr + offset + ROW_SIZE*4);
  AVX512Util::StoreReg(r5, arr + offset + ROW_SIZE*5);
  AVX512Util::StoreReg(r6, arr + offset + ROW_SIZE*6);
  AVX512Util::StoreReg(r7, arr + offset + ROW_SIZE*7);
  AVX512Util::StoreReg(r8, arr + offset + ROW_SIZE*8);
  AVX512Util::StoreReg(r9, arr + offset + ROW_SIZE*9);
  AVX512Util::StoreReg(r10, arr + offset + ROW_SIZE*10);
  AVX512Util::StoreReg(r11, arr + offset + ROW_SIZE*11);
  AVX512Util::StoreReg(r12, arr + offset + ROW_SIZE*12);
  AVX512Util::StoreReg(r13, arr + offset + ROW_SIZE*13);
  AVX512Util::StoreReg(r14, arr + offset + ROW_SIZE*14);
  AVX512Util::StoreReg(r15, arr + offset + ROW_SIZE*15);
}

template void AVX512SortUtil::SortBlock256<int, __m512i>(int *&arr, int offset);
template void AVX512SortUtil::SortBlock256<float, __m512>(float *&arr, int offset);

template <typename InType, typename RegType>
void AVX512SortUtil::SortBlock64(InType *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  RegType r0, r1, r2, r3, r4, r5, r6, r7;
  AVX512Util::LoadReg(r0, arr + offset);
  AVX512Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX512Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX512Util::LoadReg(r3, arr + offset + ROW_SIZE*3);
  AVX512Util::LoadReg(r4, arr + offset + ROW_SIZE*4);
  AVX512Util::LoadReg(r5, arr + offset + ROW_SIZE*5);
  AVX512Util::LoadReg(r6, arr + offset + ROW_SIZE*6);
  AVX512Util::LoadReg(r7, arr + offset + ROW_SIZE*7);

  // Apply bitonic sort
  AVX512Util::BitonicSort8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // transpose(shuffle) to bring in order
  AVX512Util::Transpose8x8(r0, r1, r2, r3, r4, r5, r6, r7);

  // restore into array
  AVX512Util::StoreReg(r0, &arr[offset]);
  AVX512Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX512Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX512Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
  AVX512Util::StoreReg(r4, arr + offset + ROW_SIZE*4);
  AVX512Util::StoreReg(r5, arr + offset + ROW_SIZE*5);
  AVX512Util::StoreReg(r6, arr + offset + ROW_SIZE*6);
  AVX512Util::StoreReg(r7, arr + offset + ROW_SIZE*7);
}

template void AVX512SortUtil::SortBlock64<int64_t, __m512i>(int64_t *&arr, int offset);
template void AVX512SortUtil::SortBlock64<double, __m512d>(double *&arr, int offset);

/**
 * Bitonic Sorting networks:
 * 16x16 networks: int32, float32
 * 8x8 networks: int64
 */
template <typename T>
void AVX512SortUtil::BitonicSort8x8(T &r0,
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

// 64 bit ints, doubles
template void AVX512SortUtil::BitonicSort8x8<__m512i>(__m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&);
template void AVX512SortUtil::BitonicSort8x8<__m512d>(__m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&);

template <typename T>
void AVX512SortUtil::BitonicSort16x16(T &r0, T &r1, T &r2, T &r3,
                                  T &r4, T &r5, T &r6, T &r7,
                                  T &r8, T &r9, T &r10, T &r11,
                                  T &r12, T &r13, T &r14, T &r15) {
  MinMax16(r0, r1); MinMax16(r2, r3); MinMax16(r4, r5); MinMax16(r6, r7);
  MinMax16(r8, r9); MinMax16(r10, r11); MinMax16(r12, r13); MinMax16(r14, r15);

  MinMax16(r0, r2); MinMax16(r4, r6); MinMax16(r8, r10); MinMax16(r12, r14);
  MinMax16(r1, r3); MinMax16(r5, r7); MinMax16(r9, r11); MinMax16(r13, r15);

  MinMax16(r0, r4); MinMax16(r8, r12); MinMax16(r1, r5); MinMax16(r9, r13);
  MinMax16(r2, r6); MinMax16(r10, r14); MinMax16(r3, r7); MinMax16(r11, r15);

  MinMax16(r0, r8); MinMax16(r1, r9); MinMax16(r2, r10); MinMax16(r3, r11);
  MinMax16(r4, r12); MinMax16(r5, r13); MinMax16(r6, r14); MinMax16(r7, r15);

  MinMax16(r5, r10); MinMax16(r6, r9); MinMax16(r3, r12); MinMax16(r13, r14);
  MinMax16(r7, r11); MinMax16(r1, r2); MinMax16(r4, r8);

  MinMax16(r1, r4); MinMax16(r7, r13); MinMax16(r2, r8);
  MinMax16(r11, r14); MinMax16(r5, r6); MinMax16(r9, r10);

  MinMax16(r2, r4); MinMax16(r11, r13); MinMax16(r3, r8); MinMax16(r7, r12);

  MinMax16(r6, r8); MinMax16(r10, r12); MinMax16(r3, r5); MinMax16(r7, r9);

  MinMax16(r3, r4); MinMax16(r5, r6); MinMax16(r7, r8); MinMax16(r9, r10);
  MinMax16(r11, r12);

  MinMax16(r6, r7); MinMax16(r8, r9);
}

// 32 bit ints, floats
template void AVX512SortUtil::BitonicSort16x16<__m512i>(__m512i&, __m512i&, __m512i&, __m512i&,
                                                        __m512i&, __m512i&, __m512i&, __m512i&,
                                                        __m512i&, __m512i&, __m512i&, __m512i&,
                                                        __m512i&, __m512i&, __m512i&, __m512i&);
template void AVX512SortUtil::BitonicSort16x16<__m512>(__m512&, __m512&, __m512&, __m512&,
                                                       __m512&, __m512&, __m512&, __m512&,
                                                       __m512&, __m512&, __m512&, __m512&,
                                                       __m512&, __m512&, __m512&, __m512&);


#endif

