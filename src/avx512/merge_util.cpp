#include "avx512/merge_util.h"

#ifdef AVX512

namespace avx512 {

template<typename InType, typename RegType>
void MergeRuns16(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 16;
  aligned_init(buffer, N);
  for (int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MergePass16<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MergeRuns16<int, __m512i>(int *&arr, size_t N);
template void MergeRuns16<float, __m512>(float *&arr, size_t N);

template<typename InType, typename RegType>
void MaskedMergeRuns16(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 16;
  aligned_init(buffer, N);
  for (int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MaskedMergePass16<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MaskedMergeRuns16<int, __m512i>(int *&arr, size_t N);
template void MaskedMergeRuns16<float, __m512>(float *&arr, size_t N);

template<typename InType, typename RegType>
void MergeRuns8(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 8;
  aligned_init(buffer, N);
  for (int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MergePass8<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MergeRuns8<int64_t, __m512i>(int64_t *&arr, size_t N);
template void MergeRuns8<double, __m512d>(double *&arr, size_t N);

template<typename InType, typename RegType>
void MaskedMergeRuns8(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 8;
  aligned_init(buffer, N);
  for (int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MaskedMergePass8<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MaskedMergeRuns8<int64_t, __m256i>(int64_t *&arr, size_t N);
template void MaskedMergeRuns8<double, __m256d>(double *&arr, size_t N);

template<typename InType, typename RegType>
void MergePass16(InType *&arr, InType *buffer, size_t N, int run_size) {
  int UNIT_RUN_SIZE = 16;
  RegType ra, rb;
  int buffer_offset = 0;
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge16(ra, rb);

      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if (arr[p1_ptr] > arr[p2_ptr]) {
        LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge16(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge16(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge16(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void MergePass16<int, __m512i>(int *&arr, int *buffer, size_t N, int run_size);
template void MergePass16<float, __m512>(float *&arr, float *buffer, size_t N, int run_size);

template<typename InType, typename RegType>
void MaskedMergePass16(InType *&arr, InType *buffer, size_t N, int run_size) {
  int UNIT_RUN_SIZE = 16;
  RegType ra, rb;
  int buffer_offset = 0;
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      MaskedBitonicMerge16(ra, rb);

      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if (arr[p1_ptr] > arr[p2_ptr]) {
        LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    MaskedBitonicMerge16(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge16(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge16(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void MaskedMergePass16<int, __m512i>(int *&arr, int *buffer, size_t N, int run_size);
template void MaskedMergePass16<float, __m512>(float *&arr, float *buffer, size_t N, int run_size);

template<typename InType, typename RegType>
void MergePass8(InType *&arr, InType *buffer, size_t N, int run_size) {
  int UNIT_RUN_SIZE = 8;
  RegType ra, rb;
  int buffer_offset = 0;
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge8(ra, rb);

      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if (arr[p1_ptr] > arr[p2_ptr]) {
        LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge8(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void MergePass8<int64_t, __m512i>(int64_t *&arr, int64_t *buffer, size_t N, int run_size);
template void MergePass8<double, __m512d>(double *&arr, double *buffer, size_t N, int run_size);

template<typename InType, typename RegType>
void MaskedMergePass8(InType *&arr, InType *buffer, size_t N, int run_size) {
  int UNIT_RUN_SIZE = 8;
  RegType ra, rb;
  int buffer_offset = 0;
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      MaskedBitonicMerge8(ra, rb);

      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if (arr[p1_ptr] > arr[p2_ptr]) {
        LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    MaskedBitonicMerge8(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge8(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge8(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}
template void MaskedMergePass8<int64_t, __m256i>(int64_t *&arr, int64_t *buffer, size_t N, int run_size);
template void MaskedMergePass8<double, __m256d>(double *&arr, double *buffer, size_t N, int run_size);
}

#endif