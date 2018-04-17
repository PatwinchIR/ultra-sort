#include "avx256/simd_sort.h"

#ifdef AVX2
void SIMDSorter::SIMDSort32(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    SortUtil::SortBlock64(arr, i);
  }
  // Merge sorted runs
  MergeUtil::MergeRuns8(arr, N);
}

void SIMDSorter::SIMDSort64(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 16;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    SortUtil::SortBlock16(arr, i);
  }
  // Merge sorted runs
  MergeUtil::MergeRuns4(arr, N);
}

void SIMDSorter::SIMDSort32KV(size_t N, std::pair<int, int> *&arr) {
  int64_t* kv_arr;
  aligned_init<int64_t>(kv_arr, N);
  for(int i = 0; i < N; i++) {
    kv_arr[i] = ((((int64_t)arr[i].first) << 32) | (0x00000000ffffffff & arr[i].second));
  }
  SIMDSort64(N, kv_arr);
  for(int i = 0; i < N; i++) {
    auto kv = (int*)&kv_arr[i];
    arr[i].first = kv[1];
    arr[i].second = kv[0];
  }
}
#endif