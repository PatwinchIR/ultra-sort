#include "avx512/simd_sort.h"

#ifdef AVX512
void AVX512SIMDSorter::SIMDSort(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 256;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock256<int, __m512i>(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns16<int,__m512i>(arr, N);
}

void AVX512SIMDSorter::SIMDSort(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock64<int64_t,__m512i>(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns8<int64_t,__m512i>(arr, N);
}

void AVX512SIMDSorter::SIMDSort(size_t N, float *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 256;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock256<float,__m512>(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns16<float,__m512>(arr, N);
}

void AVX512SIMDSorter::SIMDSort(size_t N, double *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock64<double,__m512d>(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns8<double,__m512d>(arr, N);
}

void AVX512SIMDSorter::SIMDSort32KV(size_t N, std::pair<int, int> *&arr) {
  int64_t* kv_arr;
  aligned_init<int64_t>(kv_arr, N);
  for(int i = 0; i < N; i++) {
    kv_arr[i] = ((((int64_t)arr[i].first) << 32) | (0x00000000ffffffff & arr[i].second));
  }
  SIMDSort(N, kv_arr);
  for(int i = 0; i < N; i++) {
    auto kv = (int*)&kv_arr[i];
    arr[i].first = kv[1];
    arr[i].second = kv[0];
  }
}

#endif
