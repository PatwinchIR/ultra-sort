#include "avx256/simd_sort.h"

#ifdef AVX2
namespace avx2 {
void SIMDSort(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock64<int, __m256i>(arr, i);
  }
  // Merge sorted runs
  MergeRuns8<int, __m256i>(arr, N);
}

void SIMDSort(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 16;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock16<int64_t, __m256i>(arr, i);
  }
  // Merge sorted runs
  MergeRuns4<int64_t, __m256i>(arr, N);
}

void SIMDSort(size_t N, float *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock64<float, __m256>(arr, i);
  }
  // Merge sorted runs
  MergeRuns8<float, __m256>(arr, N);
}

void SIMDSort(size_t N, double *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 16;
  assert(N % BLOCK_SIZE == 0);
  for (int i = 0; i < N; i += BLOCK_SIZE) {
    SortBlock16<double, __m256d>(arr, i);
  }
  // Merge sorted runs
  MergeRuns4<double, __m256d>(arr, N);
}

void SIMDSort(size_t N, std::pair<int, int> *&arr) {
  int *kv_arr;
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
    MaskedSortBlock4x8<int, __m256i>(kv_arr, i);
  }

  // Merge sorted runs
  MaskedMergeRuns8<int, __m256i>(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    arr[i].first = kv_arr[2 * i];
    arr[i].second = kv_arr[2 * i + 1];
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
  // 4 rows of 4 K-V(8 total) pairs = 32 values
  int BLOCK_SIZE = 32;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock4x8<float, __m256>(kv_arr, i);
  }

  // Merge sorted runs
  MaskedMergeRuns8<float, __m256>(kv_arr, Nkv);
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
  // 2 rows of 2 K-V(4 total) pairs = 8 values
  int BLOCK_SIZE = 8;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock2x4<int64_t, __m256i>(kv_arr, i);
  }
  // Merge sorted runs
  MaskedMergeRuns4<int64_t, __m256i>(kv_arr, Nkv);
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
  // 2 rows of 2 K-V(4 total) pairs = 8 values
  int BLOCK_SIZE = 8;
  assert(Nkv % BLOCK_SIZE == 0);
  for (int i = 0; i < Nkv; i += BLOCK_SIZE) {
    MaskedSortBlock2x4<double, __m256d>(kv_arr, i);
  }
  // Merge sorted runs
  MaskedMergeRuns4<double, __m256d>(kv_arr, Nkv);
  for (int i = 0; i < N; i++) {
    arr[i].first = kv_arr[2 * i];
    arr[i].second = kv_arr[2 * i + 1];
  }
}
}
#endif