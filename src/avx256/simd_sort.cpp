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
#endif