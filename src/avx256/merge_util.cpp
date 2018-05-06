#include "avx256/merge_util.h"

#ifdef AVX2
namespace avx2 {

template<typename InType, typename RegType>
void MergeRuns8(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 8;
  aligned_init(buffer, N);
  for (unsigned int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MergePass8<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MergeRuns8<int, __m256i>(int *&arr, size_t N);
template void MergeRuns8<float, __m256>(float *&arr, size_t N);

template<typename InType, typename RegType>
void MaskedMergeRuns8(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 8;
  aligned_init(buffer, N);
  for (unsigned int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MaskedMergePass8<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MaskedMergeRuns8<int, __m256i>(int *&arr, size_t N);
template void MaskedMergeRuns8<float, __m256>(float *&arr, size_t N);

template<typename InType, typename RegType>
void MergeRuns4(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 4;
  aligned_init(buffer, N);
  for (unsigned int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MergePass4<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}

template void MergeRuns4<int64_t, __m256i>(int64_t *&arr, size_t N);
template void MergeRuns4<double, __m256d>(double *&arr, size_t N);

template<typename InType, typename RegType>
void MaskedMergeRuns4(InType *&arr, size_t N) {
  InType *buffer;
  int UNIT_RUN_SIZE = 4;
  aligned_init(buffer, N);
  for (unsigned int run_size = UNIT_RUN_SIZE; run_size < N; run_size *= 2) {
    MaskedMergePass4<InType, RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void MaskedMergeRuns4<int64_t, __m256i>(int64_t *&arr, size_t N);
template void MaskedMergeRuns4<double, __m256d>(double *&arr, size_t N);

template<typename InType, typename RegType>
void MergePass8(InType *&arr, InType *buffer, size_t N, unsigned int run_size) {
  int UNIT_RUN_SIZE = 8;
#pragma omp parallel for
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int buffer_offset = start;
    RegType ra, rb;
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

template void MergePass8<int, __m256i>(int *&arr, int *buffer, size_t N, unsigned int run_size);
template void MergePass8<float, __m256>(float *&arr, float *buffer, size_t N, unsigned int run_size);

template<typename InType, typename RegType>
void MaskedMergePass8(InType *&arr, InType *buffer, size_t N, unsigned int run_size) {
  int UNIT_RUN_SIZE = 8;
#pragma omp parallel for
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int buffer_offset = start;
    RegType ra, rb;
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

template void MaskedMergePass8<int, __m256i>(int *&arr, int *buffer, size_t N, unsigned int run_size);
template void MaskedMergePass8<float, __m256>(float *&arr, float *buffer, size_t N, unsigned int run_size);

template<typename InType, typename RegType>
void MergePass4(InType *&arr, InType *buffer, size_t N, unsigned int run_size) {
  int UNIT_RUN_SIZE = 4;
#pragma omp parallel for
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int buffer_offset = start;
    RegType ra, rb;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge4(ra, rb);

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

    BitonicMerge4(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge4(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge4(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}
template void MergePass4<int64_t, __m256i>(int64_t *&arr, int64_t *buffer, size_t N, unsigned int run_size);
template void MergePass4<double, __m256d>(double *&arr, double *buffer, size_t N, unsigned int run_size);

template<typename InType, typename RegType>
void MaskedMergePass4(InType *&arr, InType *buffer, size_t N, unsigned int run_size) {
  int UNIT_RUN_SIZE = 4;
#pragma omp parallel for
  for (int i = 0; i < N; i += 2 * run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2 * run_size;
    int buffer_offset = start;
    RegType ra, rb;
    int p1_ptr = start;
    int p2_ptr = mid;
    LoadReg(ra, &arr[p1_ptr]);
    LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      MaskedBitonicMerge4(ra, rb);

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

    MaskedBitonicMerge4(ra, rb);

    StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge4(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      MaskedBitonicMerge4(ra, rb);
      StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}
template void MaskedMergePass4<int64_t, __m256i>(int64_t *&arr, int64_t *buffer, size_t N, unsigned int run_size);
template void MaskedMergePass4<double, __m256d>(double *&arr, double *buffer, size_t N, unsigned int run_size);
}

#endif
