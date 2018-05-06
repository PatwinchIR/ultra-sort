#include "avx512/simd_sort.h"

#ifdef AVX512

namespace avx512 {
void SIMDSort(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 256;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock256<int, __m512i>(arr, i);
  }
  // Merge sorted runs
  MergeRuns16<int, __m512i>(arr, N);
}

void SIMDSort(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock64<int64_t, __m512i>(arr, i);
  }
  // Merge sorted runs
  MergeRuns8<int64_t, __m512i>(arr, N);
}

void SIMDSort(size_t N, float *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 256;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock256<float, __m512>(arr, i);
  }
  // Merge sorted runs
  MergeRuns16<float, __m512>(arr, N);
}

void SIMDSort(size_t N, double *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock64<double, __m512d>(arr, i);
  }
  // Merge sorted runs
  MergeRuns8<double, __m512d>(arr, N);
}

void SIMDSort(size_t N, std::pair<int, int> *&arr) {
  int64_t *kv_arr;
  aligned_init<int64_t>(kv_arr, N);
  for (int i = 0; i < N; i++) {
    kv_arr[i] = ((((int64_t) arr[i].first) << 32) | (0x00000000ffffffff & arr[i].second));
  }
  SIMDSort(N, kv_arr);
  for (int i = 0; i < N; i++) {
    auto kv = (int *) &kv_arr[i];
    arr[i].first = kv[1];
    arr[i].second = kv[0];
  }
}

void SIMDOrderBy(std::pair<int, int> *&result_arr, size_t N, std::pair<int, int> *arr, int order_by) {
  int64_t *kv_arr;
  aligned_init<int64_t>(kv_arr, N);
  aligned_init<std::pair<int, int>>(result_arr, N);
  for (int i = 0; i < N; ++i) {
    auto value = (int64_t) (order_by == 0 ? arr[i].first : arr[i].second);
    kv_arr[i] = (((value) << 32) | (0x00000000ffffffff & i));
  }
  SIMDSort(N, kv_arr);
  for (int j = 0; j < N; ++j) {
    auto index = 0x00000000ffffffff & kv_arr[j];
    result_arr[j] = arr[index];
  }
}

void SIMDSort(size_t N, std::pair<float, float> *&arr) {
  float *kv_arr;
  size_t Nkv = N * 2;
  aligned_init(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    kv_arr[2 * i] = arr[i].first;
    kv_arr[2 * i + 1] = arr[i].second;
  }
  // 8 rows of 8 K-V(16 total) pairs = 128 values
  int BLOCK_SIZE = 128;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock8x16<float, __m512>(kv_arr, i);
  }

  // Merge sorted runs
  MaskedMergeRuns16<float, __m512>(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    arr[i].first = kv_arr[2 * i];
    arr[i].second = kv_arr[2 * i + 1];
  }
}

void SIMDSort(size_t N, std::pair<int64_t, int64_t> *&arr) {
  int64_t *kv_arr;
  size_t Nkv = N * 2;
  aligned_init(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    kv_arr[2 * i] = arr[i].first;
    kv_arr[2 * i + 1] = arr[i].second;
  }
  // 4 rows of 4 K-V(8 total) pairs = 32 values
  int BLOCK_SIZE = 32;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock4x8<int64_t, __m512i>(kv_arr, i);
  }
  // Merge sorted runs
  MaskedMergeRuns8<int64_t, __m512i>(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    arr[i].first = kv_arr[2 * i];
    arr[i].second = kv_arr[2 * i + 1];
  }
}

void SIMDSort(size_t N, std::pair<double, double> *&arr) {
  double *kv_arr;
  size_t Nkv = N * 2;
  aligned_init(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    kv_arr[2 * i] = arr[i].first;
    kv_arr[2 * i + 1] = arr[i].second;
  }
  // 4 rows of 4 K-V(8 total) pairs = 32 values
  int BLOCK_SIZE = 32;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock4x8<double, __m512d>(kv_arr, i);
  }
  // Merge sorted runs
  MaskedMergeRuns8<double, __m512d>(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    arr[i].first = kv_arr[2 * i];
    arr[i].second = kv_arr[2 * i + 1];
  }
}

}

#endif
