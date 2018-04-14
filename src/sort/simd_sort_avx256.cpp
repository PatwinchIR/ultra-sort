#include "sort/simd_sort_avx256.h"

#ifdef __AVX2__

void load_reg256(__m256i &r, const int* arr) {
  r = *((__m256i*)arr);
}

void store_reg256(const __m256i &r, int* arr) {
  *((__m256i*)arr) = r;
}

void minmax(__m256i &b, __m256i &a) {
  __m256i c = a;
  a = _mm256_max_epi32(a, b);
  b = _mm256_min_epi32(c, b);
}

void bitonic_sort_avx2(__m256i &r0, __m256i &r1, __m256i &r2, __m256i &r3,
                       __m256i &r4, __m256i &r5, __m256i &r6, __m256i &r7,
                       int network_size=8) {
  assert(network_size == 8);
  minmax(r0, r1);
  minmax(r2, r3);
  minmax(r4, r5);
  minmax(r6, r7);
  minmax(r0, r2);
  minmax(r4, r6);
  minmax(r1, r3);
  minmax(r5, r7);
  minmax(r1, r2);
  minmax(r5, r6);
  minmax(r0, r4);
  minmax(r1, r5);
  minmax(r1, r4);
  minmax(r2, r6);
  minmax(r3, r7);
  minmax(r3, r6);
  minmax(r2, r4);
  minmax(r3, r5);
  minmax(r3, r4);
}

void transpose8x8_ps(__m256i &row0, __m256i &row1, __m256i &row2, __m256i &row3, __m256i &row4, __m256i &row5, __m256i &row6, __m256i &row7) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps(row0, row1);
  __t1 = _mm256_unpackhi_ps(row0, row1);
  __t2 = _mm256_unpacklo_ps(row2, row3);
  __t3 = _mm256_unpackhi_ps(row2, row3);
  __t4 = _mm256_unpacklo_ps(row4, row5);
  __t5 = _mm256_unpackhi_ps(row4, row5);
  __t6 = _mm256_unpacklo_ps(row6, row7);
  __t7 = _mm256_unpackhi_ps(row6, row7);
  __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
  __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
  __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
  __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
  __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
  __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
  __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
  __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  row0 = _mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = _mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = _mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = _mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = _mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = _mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = _mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = _mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

inline __m256i reverse(__m256i& v) {
  int rev_idx_mask[8] = {7, 6, 5, 4, 3, 2, 1, 0};
  return _mm256_permutevar8x32_epi32(v, *((__m256i *) rev_idx_mask));
}

inline void minmax(const __m256i& a, const __m256i& b,
                   __m256i& minab, __m256i& maxab){
  minab = _mm256_min_epi32(a, b);
  maxab = _mm256_max_epi32(a, b);
}

inline void intra_register_sort(__m256i& a8, __m256i& b8) {
  __m256i mina, maxa, minb, maxb;
  // phase 1
  int swap_128[8] = {4, 5, 6, 7, 0, 1, 2, 3};
  auto a8_1 = _mm256_permutevar8x32_epi32(a8, *((__m256i *) swap_128));
  auto b8_1 = _mm256_permutevar8x32_epi32(b8, *((__m256i *) swap_128));

  minmax(a8, a8_1, mina, maxa);
  minmax(b8, b8_1, minb, maxb);

  auto a4 = _mm256_blend_epi32(mina, maxa, 0xf0);
  auto b4 = _mm256_blend_epi32(minb, maxb, 0xf0);

  // phase 2
  auto a4_1 = _mm256_shuffle_epi32(a4, 0x4e);
  auto b4_1 = _mm256_shuffle_epi32(b4, 0x4e);

  minmax(a4, a4_1, mina, maxa);
  minmax(b4, b4_1, minb, maxb);

  auto a2 = _mm256_unpacklo_epi64(mina, maxa);
  auto b2 = _mm256_unpacklo_epi64(minb, maxb);

  // phase 3
  auto a2_1 = _mm256_shuffle_epi32(a2, 0xb1);
  auto b2_1 = _mm256_shuffle_epi32(b2, 0xb1);

  minmax(a2, a2_1, mina, maxa);
  minmax(b2, b2_1, minb, maxb);

  a8 = _mm256_blend_epi32(mina, maxa, 0xaa);
  b8 = _mm256_blend_epi32(minb, maxb, 0xaa);
}

void bitonic_merge_avx2(__m256i& a, __m256i& b) {
  // phase 1 - 8 against 8
  b = reverse(b);
  minmax(a, b);
  intra_register_sort(a,b);
}

void sort_block_avx2(int *&arr, int start, int network_size) {
  int row_size = VECWIDTH_AVX2/NUMBITS(network_size);
  // Put into registers
  __m256i r0, r1, r2, r3, r4, r5, r6, r7;
  load_reg256(r0, arr + start);
  load_reg256(r1, arr + start + row_size);
  load_reg256(r2, arr + start + row_size*2);
  load_reg256(r3, arr + start + row_size*3);
  load_reg256(r4, arr + start + row_size*4);
  load_reg256(r5, arr + start + row_size*5);
  load_reg256(r6, arr + start + row_size*6);
  load_reg256(r7, arr + start + row_size*7);

  // Apply bitonic sort
  bitonic_sort_avx2(r0, r1, r2, r3, r4, r5, r6, r7, network_size);

  // transpose(shuffle) to bring in order
  transpose8x8_ps(r0, r1, r2, r3, r4, r5, r6, r7);

  // restore into array
  store_reg256(r0, arr + start);
  store_reg256(r1, arr + start + row_size);
  store_reg256(r2, arr + start + row_size*2);
  store_reg256(r3, arr + start + row_size*3);
  store_reg256(r4, arr + start + row_size*4);
  store_reg256(r5, arr + start + row_size*5);
  store_reg256(r6, arr + start + row_size*6);
  store_reg256(r7, arr + start + row_size*7);
}

/**
 * Merges two runs of size 'run_size' throughout the data
 * @param arr: pointer to the start of the array
 * @param buffer: buffer to flush sorted output into
 * @param N
 * @param run_size
 */
void merge_pass_avx2(int *&arr, int *buffer, int N, int run_size) {
  __m256i ra, rb;
  int network_size = 8;
  int unit_run_size = VECWIDTH_AVX2/NUMBITS(network_size);
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    load_reg256(ra, &arr[p1_ptr]);
    load_reg256(rb, &arr[p2_ptr]);
    p1_ptr += unit_run_size;
    p2_ptr += unit_run_size;

    while (p1_ptr < mid && p2_ptr < end) {
      bitonic_merge_avx2(ra, rb);

      store_reg256(ra, &buffer[buffer_offset]);
      buffer_offset += unit_run_size;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        load_reg256(ra, &arr[p2_ptr]);
        p2_ptr += unit_run_size;
      } else {
        load_reg256(ra, &arr[p1_ptr]);
        p1_ptr += unit_run_size;
      }
    }

    bitonic_merge_avx2(ra, rb);

    store_reg256(ra, &buffer[buffer_offset]);
    buffer_offset += unit_run_size;

    while (p1_ptr < mid) {
      load_reg256(ra, &arr[p1_ptr]);
      p1_ptr += unit_run_size;
      bitonic_merge_avx2(ra, rb);
      store_reg256(ra, &buffer[buffer_offset]);
      buffer_offset += unit_run_size;
    }

    while (p2_ptr < end) {
      load_reg256(ra, &arr[p2_ptr]);
      p2_ptr += unit_run_size;
      bitonic_merge_avx2(ra, rb);
      store_reg256(ra, &buffer[buffer_offset]);
      buffer_offset += unit_run_size;
    }

    store_reg256(rb, &buffer[buffer_offset]);
    buffer_offset += unit_run_size;
  }
}

void merge_runs_avx2(int *&arr, int N, int network_size) {
  int unit_run_size = VECWIDTH_AVX2/NUMBITS(network_size);
  int* buffer;
  aligned_init<int>(buffer, N);
  for(int run_size = unit_run_size; run_size < N; run_size*=2) {
    merge_pass_avx2(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}

void sort_avx2(size_t N, int *&arr, int network_size) {
  // Determine block size for the sorting network
  int block_size = network_size*(VECWIDTH_AVX2/NUMBITS(network_size));
  assert(N % block_size == 0);
  for(int i = 0; i < N; i+=block_size) {
    sort_block_avx2(arr, i, network_size);
  }

  // Merge sorted runs
  merge_runs_avx2(arr, N);

}
#endif

#ifdef __AVX512F__



    // Need to test while writing.
//    int a[8] = {1, 3, 5, 7, 9, 11, 13, 15};
//    int b[8] = {2, 4, 6, 8, 10, 12, 14, 16};
//
//    __m256i x = *((__m256i *)a);
//    __m256i y = *((__m256i *)b);
//    bitonic_merge(x, y);
#endif
