#include "avx512/merge_util.h"

#ifdef AVX512

template <typename InType, typename RegType>
void AVX512MergeUtil::MergeRuns16(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=16;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass16<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX512MergeUtil::MergeRuns16<int,__m512i>(int *&arr, int N);
template void AVX512MergeUtil::MergeRuns16<float,__m512>(float *&arr, int N);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergeRuns8(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=8;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass8<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX512MergeUtil::MergeRuns8<int64_t,__m512i>(int64_t *&arr, int N);
template void AVX512MergeUtil::MergeRuns8<double,__m512d>(double *&arr, int N);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergePass16(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=16;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX512Util::LoadReg(ra, &arr[p1_ptr]);
    AVX512Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      AVX512Util::BitonicMerge16(ra, rb);

      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX512Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX512Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    AVX512Util::BitonicMerge16(ra, rb);

    AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX512Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      AVX512Util::BitonicMerge16(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX512Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      AVX512Util::BitonicMerge16(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX512Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX512MergeUtil::MergePass16<int,__m512i>(int *&arr, int *buffer, int N, int run_size);
template void AVX512MergeUtil::MergePass16<float,__m512>(float *&arr, float *buffer, int N, int run_size);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergePass8(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=8;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX512Util::LoadReg(ra, &arr[p1_ptr]);
    AVX512Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      AVX512Util::BitonicMerge8(ra, rb);

      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX512Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX512Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    AVX512Util::BitonicMerge8(ra, rb);

    AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX512Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      AVX512Util::BitonicMerge8(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX512Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      AVX512Util::BitonicMerge8(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX512Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX512MergeUtil::MergePass8<int64_t,__m512i>(int64_t *&arr, int64_t *buffer, int N, int run_size);
template void AVX512MergeUtil::MergePass8<double,__m512d>(double *&arr, double *buffer, int N, int run_size);

#endif