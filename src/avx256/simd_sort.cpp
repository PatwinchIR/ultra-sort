#include "avx256/simd_sort.h"

#ifdef AVX2
void AVX256SIMDSorter::SIMDSort(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX256SortUtil::SortBlock64<int,__m256i>(arr, i);
  }
  // Merge sorted runs
  AVX256MergeUtil::MergeRuns8<int,__m256i>(arr, N);
}

void AVX256SIMDSorter::SIMDSort(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 16;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    SortUtil::SortBlock16<int64_t,__m256i>(arr, i);
  }
  // Merge sorted runs
  MergeUtil::MergeRuns4<int64_t,__m256i>(arr, N);
}

void AVX256SIMDSorter::SIMDSort(size_t N, float *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    SortUtil::SortBlock64<float,__m256>(arr, i);
  }
  // Merge sorted runs
  MergeUtil::MergeRuns8<float,__m256>(arr, N);
}

void AVX256SIMDSorter::SIMDSort(size_t N, double *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 16;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    SortUtil::SortBlock16<double,__m256d>(arr, i);
  }
  // Merge sorted runs
  MergeUtil::MergeRuns4<double,__m256d>(arr, N);
}

void AVX256SIMDSorter::SIMDSort32KV(size_t N, std::pair<int, int> *&arr) {
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