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

inline void minmax(const __m512i& a, const __m512i& b,
                   __m512i& minab, __m512i& maxab){
    minab = _mm512_min_epi32(a, b);
    maxab = _mm512_max_epi32(a, b);
}

void minmax(__m512i &b, __m512i &a) {
    __m512i c = a;
    a = _mm512_max_epi32(a, b);
    b = _mm512_min_epi32(c, b);
}

void transpose16x16_ps(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
                       __m512i &r4, __m512i &r5, __m512i &r6, __m512i &r7,
                       __m512i &r8, __m512i &r9, __m512i &r10, __m512i &r11,
                       __m512i &r12, __m512i &r13, __m512i &r14, __m512i &r15) {
    __m512 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
    __m512 __t8, __t9, __ta, __tb, __tc, __td, __te, __tf;

    __t0 = _mm512_unpacklo_epi32(r0,r1);   //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29 
    __t1 = _mm512_unpackhi_epi32(r0,r1);   //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
    __t2 = _mm512_unpacklo_epi32(r2,r3);   //  32  48  33  49 ...
    __t3 = _mm512_unpackhi_epi32(r2,r3);   //  34  50  35  51 ...
    __t4 = _mm512_unpacklo_epi32(r4,r5);   //  64  80  65  81 ...  
    __t5 = _mm512_unpackhi_epi32(r4,r5);   //  66  82  67  83 ...
    __t6 = _mm512_unpacklo_epi32(r6,r7);   //  96 112  97 113 ...
    __t7 = _mm512_unpackhi_epi32(r6,r7);   //  98 114  99 115 ...
    __t8 = _mm512_unpacklo_epi32(r8,r9);   // 128 ...
    __t9 = _mm512_unpackhi_epi32(r8,r9);   // 130 ...
    __ta = _mm512_unpacklo_epi32(r10,r11); // 160 ...
    __tb = _mm512_unpackhi_epi32(r10,r11); // 162 ...
    __tc = _mm512_unpacklo_epi32(r12,r13); // 196 ...
    __td = _mm512_unpackhi_epi32(r12,r13); // 198 ...
    __te = _mm512_unpacklo_epi32(r14,r15); // 228 ...
    __tf = _mm512_unpackhi_epi32(r14,r15); // 230 ...
    
    r0 = _mm512_unpacklo_epi64(__t0,__t2); //   0  16  32  48 ...
    r1 = _mm512_unpackhi_epi64(__t0,__t2); //   1  17  33  49 ...
    r2 = _mm512_unpacklo_epi64(__t1,__t3); //   2  18  34  49 ...
    r3 = _mm512_unpackhi_epi64(__t1,__t3); //   3  19  35  51 ...
    r4 = _mm512_unpacklo_epi64(__t4,__t6); //  64  80  96 112 ...  
    r5 = _mm512_unpackhi_epi64(__t4,__t6); //  65  81  97 114 ...
    r6 = _mm512_unpacklo_epi64(__t5,__t7); //  66  82  98 113 ...
    r7 = _mm512_unpackhi_epi64(__t5,__t7); //  67  83  99 115 ...
    r8 = _mm512_unpacklo_epi64(__t8,__ta); // 128 144 160 176 ...  
    r9 = _mm512_unpackhi_epi64(__t8,__ta); // 129 145 161 178 ...
    r10 = _mm512_unpacklo_epi64(__t9,__tb); // 130 146 162 177 ... 
    r11 = _mm512_unpackhi_epi64(__t9,__tb); // 131 147 163 179 ...
    r12 = _mm512_unpacklo_epi64(__tc,__te); // 192 208 228 240 ...
    r13 = _mm512_unpackhi_epi64(__tc,__te); // 193 209 229 241 ...
    r14 = _mm512_unpacklo_epi64(__td,__tf); // 194 210 230 242 ...
    r15 = _mm512_unpackhi_epi64(__td,__tf); // 195 211 231 243 ...

    __t0 = _mm512_shuffle_i32x4(r0, r4, 0x88);   //   0  16  32  48   8  24  40  56  64  80  96  112 ...
    __t1 = _mm512_shuffle_i32x4(r1, r5, 0x88);   //   1  17  33  49 ...
    __t2 = _mm512_shuffle_i32x4(r2, r6, 0x88);   //   2  18  34  50 ...
    __t3 = _mm512_shuffle_i32x4(r3, r7, 0x88);   //   3  19  35  51 ...
    __t4 = _mm512_shuffle_i32x4(r0, r4, 0xdd);   //   4  20  36  52 ...
    __t5 = _mm512_shuffle_i32x4(r1, r5, 0xdd);   //   5  21  37  53 ...
    __t6 = _mm512_shuffle_i32x4(r2, r6, 0xdd);   //   6  22  38  54 ...
    __t7 = _mm512_shuffle_i32x4(r3, r7, 0xdd);   //   7  23  39  55 ...
    __t8 = _mm512_shuffle_i32x4(r8, r12, 0x88);  // 128 144 160 176 ...
    __t9 = _mm512_shuffle_i32x4(r9, r13, 0x88);  // 129 145 161 177 ...
    __ta = _mm512_shuffle_i32x4(r10, r14, 0x88); // 130 146 162 178 ...
    __tb = _mm512_shuffle_i32x4(r11, r15, 0x88); // 131 147 163 179 ...
    __tc = _mm512_shuffle_i32x4(r8, r12, 0xdd);  // 132 148 164 180 ...
    __td = _mm512_shuffle_i32x4(r9, r13, 0xdd);  // 133 149 165 181 ...
    __te = _mm512_shuffle_i32x4(r10, r14, 0xdd); // 134 150 166 182 ...
    __tf = _mm512_shuffle_i32x4(r11, r15, 0xdd); // 135 151 167 183 ...

    r0 = _mm512_shuffle_i32x4(__t0, __t8, 0x88);  //   0  16  32  48  64  80  96 112 ... 240
    r1 = _mm512_shuffle_i32x4(__t1, __t9, 0x88);  //   1  17  33  49  66  81  97 113 ... 241
    r2 = _mm512_shuffle_i32x4(__t2, __ta, 0x88);  //   2  18  34  50  67  82  98 114 ... 242
    r3 = _mm512_shuffle_i32x4(__t3, __tb, 0x88);  //   3  19  35  51  68  83  99 115 ... 243
    r4 = _mm512_shuffle_i32x4(__t4, __tc, 0x88);  //   4 ...
    r5 = _mm512_shuffle_i32x4(__t5, __td, 0x88);  //   5 ...
    r6 = _mm512_shuffle_i32x4(__t6, __te, 0x88);  //   6 ...
    r7 = _mm512_shuffle_i32x4(__t7, __tf, 0x88);  //   7 ...
    r8 = _mm512_shuffle_i32x4(__t0, __t8, 0xdd);  //   8 ...
    r9 = _mm512_shuffle_i32x4(__t1, __t9, 0xdd);  //   9 ...
    r10 = _mm512_shuffle_i32x4(__t2, __ta, 0xdd); //  10 ...
    r11 = _mm512_shuffle_i32x4(__t3, __tb, 0xdd); //  11 ...
    r12 = _mm512_shuffle_i32x4(__t4, __tc, 0xdd); //  12 ...
    r13 = _mm512_shuffle_i32x4(__t5, __td, 0xdd); //  13 ...
    r14 = _mm512_shuffle_i32x4(__t6, __te, 0xdd); //  14 ...
    r15 = _mm512_shuffle_i32x4(__t7, __tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}

inline __m512i reverse(__m512i& v) {
    int rev_idx_mask[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    return _mm512_permutevar8x32_epi32(v, *((__m512i *) rev_idx_mask));
}

inline void bitonic_merge(__m512i& a, __512i& b) {
    // phase 1 - 16 against 16
    b = reverse(b);
    minmax(a, b);
    intra_register_sort(a,b);
}

void bitonic_sort_avx2(__m512i &r0, __m512i &r1, __m512i &r2, __m512i &r3,
                       __m512i &r4, __m512i &r5, __m512i &r6, __m512i &r7,
                       __m512i &r8, __m512i &r9, __m512i &r10, __m512i &r11,
                       __m512i &r12, __m512i &r13, __m512i &r14, __m512i &r15,
                       int network_size=16) {
    assert(network_size == 16);

    minmax(r0, r1); minmax(r2, r3); minmax(r4, r5); minmax(r6, r7);
    minmax(r8, r9); minmax(r10, r11); minmax(r12, r13); minmax(r14, r15);

    minmax(r0, r2); minmax(r4, r6); minmax(r8, r10); minmax(r12, r14);
    minmax(r1, r3); minmax(r5, r7); minmax(r9, r11); minmax(r13, r15);

    minmax(r0, r4); minmax(r8, r12); minmax(r1, r5); minmax(r9, r13);
    minmax(r2, r6); minmax(r10, r14); minmax(r3, r7); minmax(r11, r15);

    minmax(r0, r8); minmax(r1, r9); minmax(r2, r10); minmax(r3, r11);
    minmax(r4, r12); minmax(r5, r13); minmax(r6, r14); minmax(r7, r15);

    minmax(r5, r10); minmax(r6, r9); minmax(r3, r12); minmax(r13, r14);
    minmax(r7, r11); minmax(r1, r2); minmax(r4, r8);

    minmax(r1, r4); minmax(r7, r13); minmax(r2, r8); minmax(r11, r14);
    minmax(r5, r6); minmax(r9, r10);

    minmax(r2, r4); minmax(r11, r13); minmax(r3, r8); minmax(r7, r12);

    minmax(r6, r8); minmax(r10, r12); minmax(r3, r5); minmax(r7, r9);

    minmax(r3, r4); minmax(r5, r6); minmax(r7, r8); minmax(r9, r10);
    minmax(r11, r12);

    minmax(r6, r7); minmax(r8, r9);
}

void sort_block_avx512(int *arr, int start, int network_size) {
    __m512i r0, r1, r2,  r3,  r4,  r5,  r6,  r7;
    __m512i r8, r9, r10, r11, r12, r13, r14, r15;
    r0 = ((__m512i*)&arr[start])[0];
    r1 = ((__m512i*)&arr[start])[1];
    r2 = ((__m512i*)&arr[start])[2];
    r3 = ((__m512i*)&arr[start])[3];
    r4 = ((__m512i*)&arr[start])[4];
    r5 = ((__m512i*)&arr[start])[5];
    r6 = ((__m512i*)&arr[start])[6];
    r7 = ((__m512i*)&arr[start])[7];
    r8 = ((__m512i*)&arr[start])[8];
    r9 = ((__m512i*)&arr[start])[9];
    r10 = ((__m512i*)&arr[start])[10];
    r11 = ((__m512i*)&arr[start])[11];
    r12 = ((__m512i*)&arr[start])[12];
    r13 = ((__m512i*)&arr[start])[13];
    r14 = ((__m512i*)&arr[start])[14];
    r15 = ((__m512i*)&arr[start])[15];
    bitonic_sort_avx512(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, network_size);

    transpose16x16_ps(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15);

    // Need to test while writing.
//    int a[8] = {1, 3, 5, 7, 9, 11, 13, 15};
//    int b[8] = {2, 4, 6, 8, 10, 12, 14, 16};
//
//    __m256i x = *((__m256i *)a);
//    __m256i y = *((__m256i *)b);
//    bitonic_merge(x, y);
}
#endif
