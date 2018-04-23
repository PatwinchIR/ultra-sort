#include "avx256/merge_util.h"

#ifdef AVX2

template <typename InType, typename RegType>
void AVX256MergeUtil::MergeRuns8(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=8;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass8<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX256MergeUtil::MergeRuns8<int,__m256i>(int *&arr, int N);
template void AVX256MergeUtil::MergeRuns8<float,__m256>(float *&arr, int N);

template <typename InType, typename RegType>
void AVX256MergeUtil::MergeRuns4(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=4;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass4<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}

template void AVX256MergeUtil::MergeRuns4<int64_t,__m256i>(int64_t *&arr, int N);
template void AVX256MergeUtil::MergeRuns4<double,__m256d>(double *&arr, int N);

template <typename InType, typename RegType>
void AVX256MergeUtil::MergePass8(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=8;
  RegType ra, rb;
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

template void AVX256MergeUtil::MergePass8<int,__m256i>(int *&arr, int *buffer, int N, int run_size);
template void AVX256MergeUtil::MergePass8<float,__m256>(float *&arr, float *buffer, int N, int run_size);

template <typename InType, typename RegType>
void AVX256Util::MergePass4(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=4;
  RegType ra, rb;
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
      AVX256Util::BitonicMerge4(ra, rb);

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

    AVX256Util::BitonicMerge4(ra, rb);

    AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX256Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      AVX256Util::BitonicMerge4(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX256Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      AVX256Util::BitonicMerge4(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX256Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}


template void AVX256MergeUtil::MergePass4<int64_t,__m256i>(int64_t *&arr, int64_t *buffer, int N, int run_size);
template void AVX256MergeUtil::MergePass4<double,__m256d>(double *&arr, double *buffer, int N, int run_size);

#endif
