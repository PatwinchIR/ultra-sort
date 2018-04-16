#include "avx256/sort_util.h"

#ifdef AVX2

void SortUtil::SortBlock64(int *&arr, int offset) {
  int ROW_SIZE = 8;
  // Put into registers
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  AVX256Util::LoadReg(r0, arr + offset);
  AVX256Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::LoadReg(r3, arr + offset + ROW_SIZE*3);
  AVX256Util::LoadReg(r4, arr + offset + ROW_SIZE*4);
  AVX256Util::LoadReg(r5, arr + offset + ROW_SIZE*5);
  AVX256Util::LoadReg(r6, arr + offset + ROW_SIZE*6);
  AVX256Util::LoadReg(r7, arr + offset + ROW_SIZE*7);

  // Apply bitonic sort
  AVX256Util::BitonicSort8x8(r0, r1, r2, r3, r4, r5, r6, r7);

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

void SortUtil::SortBlock16(int64_t *&arr, int offset) {
  int ROW_SIZE = 4;
  // Put into registers
  __m256i r0, r1, r2, r3;
  AVX256Util::LoadReg(r0, arr + offset);
  AVX256Util::LoadReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::LoadReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::LoadReg(r3, arr + offset + ROW_SIZE*3);

  // Apply bitonic sort
  AVX256Util::BitonicSort4x4(r0, r1, r2, r3);

  // transpose(shuffle) to bring in order
  AVX256Util::Transpose4x4(r0, r1, r2, r3);

  // restore into array
  AVX256Util::StoreReg(r0, &arr[offset]);
  AVX256Util::StoreReg(r1, arr + offset + ROW_SIZE);
  AVX256Util::StoreReg(r2, arr + offset + ROW_SIZE*2);
  AVX256Util::StoreReg(r3, arr + offset + ROW_SIZE*3);
}

#endif

