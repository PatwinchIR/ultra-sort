#include "avx512/sort_util.h"

#ifdef AVX512

void AVX512SortUtil::SortBlock256(int *&arr, int offset) {
  int ROW_SIZE = 16;
  // Put into registers
  __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
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

void AVX256SortUtil::SortBlock64(int64_t *&arr, int offset) {
  int ROW_SIZE = 8;
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

