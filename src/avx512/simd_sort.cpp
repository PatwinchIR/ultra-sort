#include "avx512/simd_sort.h"

#ifdef AVX512
void AVX512SIMDSorter::SIMDSort32(size_t N, int *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 256;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock256(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns16(arr, N);
}

void AVX512SIMDSorter::SIMDSort64(size_t N, int64_t *&arr) {
  // Determine block size for the sorting network
  int BLOCK_SIZE = 64;
  assert(N % BLOCK_SIZE == 0);
  for(int i = 0; i < N; i+=BLOCK_SIZE) {
    AVX512SortUtil::SortBlock64(arr, i);
  }
  // Merge sorted runs
  AVX512MergeUtil::MergeRuns8(arr, N);
}
#endif
