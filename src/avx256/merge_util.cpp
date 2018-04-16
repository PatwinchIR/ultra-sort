#include "avx256/merge_util.h"

#ifdef AVX2

void MergeUtil::MergeRuns8(int *&arr, int N) {
  int* buffer;
  int UNIT_RUN_SIZE=8;
  aligned_init<int>(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass8(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}

void MergeUtil::MergePass8(int *&arr, int *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=8;
  __m256i ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX256Util::LoadReg(ra, &arr[p1_ptr]);
    AVX256Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      AVX256Util::BitonicMerge8(ra, rb);

      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX256Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX256Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    AVX256Util::BitonicMerge8(ra, rb);

    AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX256Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      AVX256Util::BitonicMerge8(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX256Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      AVX256Util::BitonicMerge8(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX256Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

#endif
