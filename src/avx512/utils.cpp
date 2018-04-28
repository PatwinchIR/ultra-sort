#include "avx512/utils.h"
#include "common.h"

#ifdef AVX512

/**
 * Load and Store Instructions
 */

template <typename InType, typename RegType>
void AVX512Util::LoadReg(RegType &r, InType *arr) {
  r = *((RegType*)arr);
}

template void AVX512Util::LoadReg<int, __m512i>( __m512i &r, int *arr);
template void AVX512Util::LoadReg<int64_t, __m512i>(__m512i &r, int64_t *arr);
template void AVX512Util::LoadReg<float, __m512>(__m512 &r, float *arr);
template void AVX512Util::LoadReg<double, __m512d>(__m512d &r, double *arr);

template <typename InType, typename RegType>
void AVX512Util::StoreReg(const RegType &r, InType *arr) {
  *((RegType*)arr) = r;
}

template void AVX512Util::StoreReg<int, __m512i>(const __m512i &r, int *arr);
template void AVX512Util::StoreReg<int64_t, __m512i>(const __m512i &r, int64_t *arr);
template void AVX512Util::StoreReg<float, __m512>(const __m512 &r, float *arr);
template void AVX512Util::StoreReg<double, __m512d>(const __m512d &r, double *arr);

// Weird Converter


/**
 * MinMax functions
 */

void AVX512Util::MinMax16(__m512i &a, __m512i &b) {
  __m512i c = a;
  a = _mm512_min_epi32(a, b);
  b = _mm512_max_epi32(c, b);
}

void AVX512Util::MinMax16(const __m512i& a, const __m512i& b,
                          __m512i& minab, __m512i& maxab) {
  minab = _mm512_min_epi32(a, b);
  maxab = _mm512_max_epi32(a, b);
}

void AVX512Util::MinMax16(__m512 &a, __m512 &b) {
  __m512 c = a;
  a = _mm512_min_ps(a, b);
  b = _mm512_max_ps(c, b);
}

void AVX512Util::MinMax16(const __m512& a, const __m512& b,
                          __m512& minab, __m512& maxab) {
  minab = _mm512_min_ps(a, b);
  maxab = _mm512_max_ps(a, b);
}

void AVX512Util::MinMax8(__m512i &a, __m512i &b) {
  __m512i c = a;
  a = _mm512_min_epi64(a, b);
  b = _mm512_max_epi64(c, b);
}

void AVX512Util::MinMax8(const __m512i &a, const __m512i &b,
                         __m512i& minab, __m512i& maxab) {
  minab = _mm512_min_epi64(a, b);
  maxab = _mm512_max_epi64(a, b);
}

void AVX512Util::MinMax8(__m512d &a, __m512d &b) {
  __m512d c = a;
  a = _mm512_min_pd(a, b);
  b = _mm512_max_pd(c, b);
}

void AVX512Util::MinMax8(const __m512d &a, const __m512d &b,
                         __m512d& minab, __m512d& maxab) {
  minab = _mm512_min_pd(a, b);
  maxab = _mm512_max_pd(a, b);
}

/**
 * Bitonic Sorting networks:
 * 16x16 networks: int32, float32
 * 8x8 networks: int64
 */
template <typename T>
void AVX512Util::BitonicSort8x8(T &r0,
                                T &r1,
                                T &r2,
                                T &r3,
                                T &r4,
                                T &r5,
                                T &r6,
                                T &r7) {
  MinMax8(r0, r1);
  MinMax8(r2, r3);
  MinMax8(r4, r5);
  MinMax8(r6, r7);
  MinMax8(r0, r2);
  MinMax8(r4, r6);
  MinMax8(r1, r3);
  MinMax8(r5, r7);
  MinMax8(r1, r2);
  MinMax8(r5, r6);
  MinMax8(r0, r4);
  MinMax8(r1, r5);
  MinMax8(r1, r4);
  MinMax8(r2, r6);
  MinMax8(r3, r7);
  MinMax8(r3, r6);
  MinMax8(r2, r4);
  MinMax8(r3, r5);
  MinMax8(r3, r4);
}

// 64 bit ints, doubles
template void AVX512Util::BitonicSort8x8<__m512i>(__m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&, __m512i&);
template void AVX512Util::BitonicSort8x8<__m512d>(__m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&, __m512d&);

template <typename T>
void AVX512Util::BitonicSort16x16(T &r0, T &r1, T &r2, T &r3,
                                  T &r4, T &r5, T &r6, T &r7,
                                  T &r8, T &r9, T &r10, T &r11,
                                  T &r12, T &r13, T &r14, T &r15) {
  MinMax16(r0, r1); MinMax16(r2, r3); MinMax16(r4, r5); MinMax16(r6, r7);
  MinMax16(r8, r9); MinMax16(r10, r11); MinMax16(r12, r13); MinMax16(r14, r15);

  MinMax16(r0, r2); MinMax16(r4, r6); MinMax16(r8, r10); MinMax16(r12, r14);
  MinMax16(r1, r3); MinMax16(r5, r7); MinMax16(r9, r11); MinMax16(r13, r15);

  MinMax16(r0, r4); MinMax16(r8, r12); MinMax16(r1, r5); MinMax16(r9, r13);
  MinMax16(r2, r6); MinMax16(r10, r14); MinMax16(r3, r7); MinMax16(r11, r15);

  MinMax16(r0, r8); MinMax16(r1, r9); MinMax16(r2, r10); MinMax16(r3, r11);
  MinMax16(r4, r12); MinMax16(r5, r13); MinMax16(r6, r14); MinMax16(r7, r15);

  MinMax16(r5, r10); MinMax16(r6, r9); MinMax16(r3, r12); MinMax16(r13, r14);
  MinMax16(r7, r11); MinMax16(r1, r2); MinMax16(r4, r8);

  MinMax16(r1, r4); MinMax16(r7, r13); MinMax16(r2, r8);
  MinMax16(r11, r14); MinMax16(r5, r6); MinMax16(r9, r10);

  MinMax16(r2, r4); MinMax16(r11, r13); MinMax16(r3, r8); MinMax16(r7, r12);

  MinMax16(r6, r8); MinMax16(r10, r12); MinMax16(r3, r5); MinMax16(r7, r9);

  MinMax16(r3, r4); MinMax16(r5, r6); MinMax16(r7, r8); MinMax16(r9, r10);
  MinMax16(r11, r12);
  
  MinMax16(r6, r7); MinMax16(r8, r9);
}

// 32 bit ints, floats
template void AVX512Util::BitonicSort16x16<__m512i>(__m512i&, __m512i&, __m512i&, __m512i&,
                                                    __m512i&, __m512i&, __m512i&, __m512i&,
                                                    __m512i&, __m512i&, __m512i&, __m512i&,
                                                    __m512i&, __m512i&, __m512i&, __m512i&);
template void AVX512Util::BitonicSort16x16<__m512>(__m512&, __m512&, __m512&, __m512&,
                                                   __m512&, __m512&, __m512&, __m512&,
                                                   __m512&, __m512&, __m512&, __m512&,
                                                   __m512&, __m512&, __m512&, __m512&);

/**
 * Bitonic Transpose:
 * 16x16 networks: int32, float32
 * 8x8 networks: int64
 */
void AVX512Util::Transpose8x8(__m512i &row0,
                              __m512i &row1,
                              __m512i &row2,
                              __m512i &row3,
                              __m512i &row4,
                              __m512i &row5,
                              __m512i &row6,
                              __m512i &row7) {
  // TODO: Convert
//  __m512i __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
//  __m512i __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
//  __t0 = _mm512_unpacklo_epi64(row0, row1);
//  __t1 = _mm512_unpackhi_epi64(row0, row1);
//  __t2 = _mm512_unpacklo_epi64(row2, row3);
//  __t3 = _mm512_unpackhi_epi64(row2, row3);
//  __t4 = _mm512_unpacklo_epi64(row4, row5);
//  __t5 = _mm512_unpackhi_epi64(row4, row5);
//  __t6 = _mm512_unpacklo_epi64(row6, row7);
//  __t7 = _mm512_unpackhi_epi64(row6, row7);
//  // Note: https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
//  // As provided in above link, it is relatively safe/free to run float shuffles on integers(not other way around)
//  __tt0 = _mm512_shuffle_pd(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
//  __tt1 = _mm512_shuffle_pd(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
//  __tt2 = _mm512_shuffle_pd(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
//  __tt3 = _mm512_shuffle_pd(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
//  __tt4 = _mm512_shuffle_pd(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
//  __tt5 = _mm512_shuffle_pd(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
//  __tt6 = _mm512_shuffle_pd(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
//  __tt7 = _mm512_shuffle_pd(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
//  row0 = (__m512i)_mm512_permute2f128_ps(__tt0, __tt4, 0x20);
//  row1 = (__m512i)_mm512_permute2f128_ps(__tt1, __tt5, 0x20);
//  row2 = (__m512i)_mm512_permute2f128_ps(__tt2, __tt6, 0x20);
//  row3 = (__m512i)_mm512_permute2f128_ps(__tt3, __tt7, 0x20);
//  row4 = (__m512i)_mm512_permute2f128_ps(__tt0, __tt4, 0x31);
//  row5 = (__m512i)_mm512_permute2f128_ps(__tt1, __tt5, 0x31);
//  row6 = (__m512i)_mm512_permute2f128_ps(__tt2, __tt6, 0x31);
//  row7 = (__m512i)_mm512_permute2f128_ps(__tt3, __tt7, 0x31);
}

void AVX512Util::Transpose8x8(__m512d &row0,
                              __m512d &row1,
                              __m512d &row2,
                              __m512d &row3,
                              __m512d &row4,
                              __m512d &row5,
                              __m512d &row6,
                              __m512d &row7) {
  // TODO: Convert
//  __m512 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
//  __m512 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
//  __t0 = _mm512_unpacklo_ps(row0, row1);
//  __t1 = _mm512_unpackhi_ps(row0, row1);
//  __t2 = _mm512_unpacklo_ps(row2, row3);
//  __t3 = _mm512_unpackhi_ps(row2, row3);
//  __t4 = _mm512_unpacklo_ps(row4, row5);
//  __t5 = _mm512_unpackhi_ps(row4, row5);
//  __t6 = _mm512_unpacklo_ps(row6, row7);
//  __t7 = _mm512_unpackhi_ps(row6, row7);
//  // Note: https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
//  // As provided in above link, it is relatively safe/free to run float shuffles on integers(not other way around)
//  __tt0 = _mm512_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
//  __tt1 = _mm512_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
//  __tt2 = _mm512_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
//  __tt3 = _mm512_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
//  __tt4 = _mm512_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
//  __tt5 = _mm512_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
//  __tt6 = _mm512_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
//  __tt7 = _mm512_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
//  row0 = _mm512_permute2f128_ps(__tt0, __tt4, 0x20);
//  row1 = _mm512_permute2f128_ps(__tt1, __tt5, 0x20);
//  row2 = _mm512_permute2f128_ps(__tt2, __tt6, 0x20);
//  row3 = _mm512_permute2f128_ps(__tt3, __tt7, 0x20);
//  row4 = _mm512_permute2f128_ps(__tt0, __tt4, 0x31);
//  row5 = _mm512_permute2f128_ps(__tt1, __tt5, 0x31);
//  row6 = _mm512_permute2f128_ps(__tt2, __tt6, 0x31);
//  row7 = _mm512_permute2f128_ps(__tt3, __tt7, 0x31);
}

void AVX512Util::Transpose16x16(__m512i &row0, __m512i &row1, __m512i &row2, __m512i &row3,
                    __m512i &row4, __m512i &row5, __m512i &row6, __m512i &row7,
                    __m512i &row8, __m512i &row9, __m512i &row10, __m512i &row11,
                    __m512i &row12, __m512i &row13, __m512i &row14, __m512i &row15) {
  __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  __m512i tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tta, ttb, ttc, ttd, tte, ttf;
  t0 = _mm512_unpacklo_epi32(row0,row1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
  t1 = _mm512_unpackhi_epi32(row0,row1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t2 = _mm512_unpacklo_epi32(row2,row3); //  32  48  33  49 ...
  t3 = _mm512_unpackhi_epi32(row2,row3); //  34  50  35  51 ...
  t4 = _mm512_unpacklo_epi32(row4,row5); //  64  80  65  81 ...
  t5 = _mm512_unpackhi_epi32(row4,row5); //  66  82  67  83 ...
  t6 = _mm512_unpacklo_epi32(row6,row7); //  96 112  97 113 ...
  t7 = _mm512_unpackhi_epi32(row6,row7); //  98 114  99 115 ...
  t8 = _mm512_unpacklo_epi32(row8,row9); // 128 ...
  t9 = _mm512_unpackhi_epi32(row8,row9); // 130 ...
  ta = _mm512_unpacklo_epi32(row10,row11); // 160 ...
  tb = _mm512_unpackhi_epi32(row10,row11); // 162 ...
  tc = _mm512_unpacklo_epi32(row12,row13); // 196 ...
  td = _mm512_unpackhi_epi32(row12,row13); // 198 ...
  te = _mm512_unpacklo_epi32(row14,row15); // 228 ...
  tf = _mm512_unpackhi_epi32(row14,row15); // 230 ...

  tt0 = _mm512_unpacklo_epi64(t0,t2); //   0  16  32  48 ...
  tt1 = _mm512_unpackhi_epi64(t0,t2); //   1  17  33  49 ...
  tt2 = _mm512_unpacklo_epi64(t1,t3); //   2  18  34  49 ...
  tt3 = _mm512_unpackhi_epi64(t1,t3); //   3  19  35  51 ...
  tt4 = _mm512_unpacklo_epi64(t4,t6); //  64  80  96 112 ...
  tt5 = _mm512_unpackhi_epi64(t4,t6); //  65  81  97 114 ...
  tt6 = _mm512_unpacklo_epi64(t5,t7); //  66  82  98 113 ...
  tt7 = _mm512_unpackhi_epi64(t5,t7); //  67  83  99 115 ...
  tt8 = _mm512_unpacklo_epi64(t8,ta); // 128 144 160 176 ...
  tt9 = _mm512_unpackhi_epi64(t8,ta); // 129 145 161 178 ...
  tta = _mm512_unpacklo_epi64(t9,tb); // 130 146 162 177 ...
  ttb = _mm512_unpackhi_epi64(t9,tb); // 131 147 163 179 ...
  ttc = _mm512_unpacklo_epi64(tc,te); // 192 208 228 240 ...
  ttd = _mm512_unpackhi_epi64(tc,te); // 193 209 229 241 ...
  tte = _mm512_unpacklo_epi64(td,tf); // 194 210 230 242 ...
  ttf = _mm512_unpackhi_epi64(td,tf); // 195 211 231 243 ...

  t0 = _mm512_shuffle_i32x4(tt0, tt4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t1 = _mm512_shuffle_i32x4(tt1, tt5, 0x88); //   1  17  33  49 ...
  t2 = _mm512_shuffle_i32x4(tt2, tt6, 0x88); //   2  18  34  50 ...
  t3 = _mm512_shuffle_i32x4(tt3, tt7, 0x88); //   3  19  35  51 ...
  t4 = _mm512_shuffle_i32x4(tt0, tt4, 0xdd); //   4  20  36  52 ...
  t5 = _mm512_shuffle_i32x4(tt1, tt5, 0xdd); //   5  21  37  53 ...
  t6 = _mm512_shuffle_i32x4(tt2, tt6, 0xdd); //   6  22  38  54 ...
  t7 = _mm512_shuffle_i32x4(tt3, tt7, 0xdd); //   7  23  39  55 ...
  t8 = _mm512_shuffle_i32x4(tt8, ttc, 0x88); // 128 144 160 176 ...
  t9 = _mm512_shuffle_i32x4(tt9, ttd, 0x88); // 129 145 161 177 ...
  ta = _mm512_shuffle_i32x4(tta, tte, 0x88); // 130 146 162 178 ...
  tb = _mm512_shuffle_i32x4(ttb, ttf, 0x88); // 131 147 163 179 ...
  tc = _mm512_shuffle_i32x4(tt8, ttc, 0xdd); // 132 148 164 180 ...
  td = _mm512_shuffle_i32x4(tt9, ttd, 0xdd); // 133 149 165 181 ...
  te = _mm512_shuffle_i32x4(tta, tte, 0xdd); // 134 150 166 182 ...
  tf = _mm512_shuffle_i32x4(ttb, ttf, 0xdd); // 135 151 167 183 ...

  row0 = _mm512_shuffle_i32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
  row1 = _mm512_shuffle_i32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
  row2 = _mm512_shuffle_i32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
  row3 = _mm512_shuffle_i32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
  row4 = _mm512_shuffle_i32x4(t4, tc, 0x88); //   4 ...
  row5 = _mm512_shuffle_i32x4(t5, td, 0x88); //   5 ...
  row6 = _mm512_shuffle_i32x4(t6, te, 0x88); //   6 ...
  row7 = _mm512_shuffle_i32x4(t7, tf, 0x88); //   7 ...
  row8 = _mm512_shuffle_i32x4(t0, t8, 0xdd); //   8 ...
  row9 = _mm512_shuffle_i32x4(t1, t9, 0xdd); //   9 ...
  row10 = _mm512_shuffle_i32x4(t2, ta, 0xdd); //  10 ...
  row11 = _mm512_shuffle_i32x4(t3, tb, 0xdd); //  11 ...
  row12 = _mm512_shuffle_i32x4(t4, tc, 0xdd); //  12 ...
  row13 = _mm512_shuffle_i32x4(t5, td, 0xdd); //  13 ...
  row14 = _mm512_shuffle_i32x4(t6, te, 0xdd); //  14 ...
  row15 = _mm512_shuffle_i32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}

void AVX512Util::Transpose16x16(__m512 &row0, __m512 &row1, __m512 &row2, __m512 &row3,
                    __m512 &row4, __m512 &row5, __m512 &row6, __m512 &row7,
                    __m512 &row8, __m512 &row9, __m512 &row10, __m512 &row11,
                    __m512 &row12, __m512 &row13, __m512 &row14, __m512 &row15) {
  __m512 t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
  __m512d tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9, tta, ttb, ttc, ttd, tte, ttf;
  // Note: The unpack section is converted to integer to void compiler error on Dev5.
  t0 = _mm512_unpacklo_ps(row0,row1); //   0  16   1  17   4  20   5  21   8  24   9  25  12  28  13  29
  t1 = _mm512_unpackhi_ps(row0,row1); //   2  18   3  19   6  22   7  23  10  26  11  27  14  30  15  31
  t2 = _mm512_unpacklo_ps(row2,row3); //  32  48  33  49 ...
  t3 = _mm512_unpackhi_ps(row2,row3); //  34  50  35  51 ...
  t4 = _mm512_unpacklo_ps(row4,row5); //  64  80  65  81 ...
  t5 = _mm512_unpackhi_ps(row4,row5); //  66  82  67  83 ...
  t6 = _mm512_unpacklo_ps(row6,row7); //  96 112  97 113 ...
  t7 = _mm512_unpackhi_ps(row6,row7); //  98 114  99 115 ...
  t8 = _mm512_unpacklo_ps(row8,row9); // 128 ...
  t9 = _mm512_unpackhi_ps(row8,row9); // 130 ...
  ta = _mm512_unpacklo_ps(row10,row11); // 160 ...
  tb = _mm512_unpackhi_ps(row10,row11); // 162 ...
  tc = _mm512_unpacklo_ps(row12,row13); // 196 ...
  td = _mm512_unpackhi_ps(row12,row13); // 198 ...
  te = _mm512_unpacklo_ps(row14,row15); // 228 ...
  tf = _mm512_unpackhi_ps(row14,row15); // 230 ...

  tt0 = _mm512_unpacklo_pd((__m512d)t0,(__m512d)t2); //   0  16  32  48 ...
  tt1 = _mm512_unpackhi_pd((__m512d)t0,(__m512d)t2); //   1  17  33  49 ...
  tt2 = _mm512_unpacklo_pd((__m512d)t1,(__m512d)t3); //   2  18  34  49 ...
  tt3 = _mm512_unpackhi_pd((__m512d)t1,(__m512d)t3); //   3  19  35  51 ...
  tt4 = _mm512_unpacklo_pd((__m512d)t4,(__m512d)t6); //  64  80  96 112 ...
  tt5 = _mm512_unpackhi_pd((__m512d)t4,(__m512d)t6); //  65  81  97 114 ...
  tt6 = _mm512_unpacklo_pd((__m512d)t5,(__m512d)t7); //  66  82  98 113 ...
  tt7 = _mm512_unpackhi_pd((__m512d)t5,(__m512d)t7); //  67  83  99 115 ...
  tt8 = _mm512_unpacklo_pd((__m512d)t8,(__m512d)ta); // 128 144 160 176 ...
  tt9 = _mm512_unpackhi_pd((__m512d)t8,(__m512d)ta); // 129 145 161 178 ...
  tta = _mm512_unpacklo_pd((__m512d)t9,(__m512d)tb); // 130 146 162 177 ...
  ttb = _mm512_unpackhi_pd((__m512d)t9,(__m512d)tb); // 131 147 163 179 ...
  ttc = _mm512_unpacklo_pd((__m512d)tc,(__m512d)te); // 192 208 228 240 ...
  ttd = _mm512_unpackhi_pd((__m512d)tc,(__m512d)te); // 193 209 229 241 ...
  tte = _mm512_unpacklo_pd((__m512d)td,(__m512d)tf); // 194 210 230 242 ...
  ttf = _mm512_unpackhi_pd((__m512d)td,(__m512d)tf); // 195 211 231 243 ...

  t0 = _mm512_shuffle_f32x4((__m512)tt0, (__m512)tt4, 0x88); //   0  16  32  48   8  24  40  56  64  80  96  112 ...
  t1 = _mm512_shuffle_f32x4((__m512)tt1, (__m512)tt5, 0x88); //   1  17  33  49 ...
  t2 = _mm512_shuffle_f32x4((__m512)tt2, (__m512)tt6, 0x88); //   2  18  34  50 ...
  t3 = _mm512_shuffle_f32x4((__m512)tt3, (__m512)tt7, 0x88); //   3  19  35  51 ...
  t4 = _mm512_shuffle_f32x4((__m512)tt0, (__m512)tt4, 0xdd); //   4  20  36  52 ...
  t5 = _mm512_shuffle_f32x4((__m512)tt1, (__m512)tt5, 0xdd); //   5  21  37  53 ...
  t6 = _mm512_shuffle_f32x4((__m512)tt2, (__m512)tt6, 0xdd); //   6  22  38  54 ...
  t7 = _mm512_shuffle_f32x4((__m512)tt3, (__m512)tt7, 0xdd); //   7  23  39  55 ...
  t8 = _mm512_shuffle_f32x4((__m512)tt8, (__m512)ttc, 0x88); // 128 144 160 176 ...
  t9 = _mm512_shuffle_f32x4((__m512)tt9, (__m512)ttd, 0x88); // 129 145 161 177 ...
  ta = _mm512_shuffle_f32x4((__m512)tta, (__m512)tte, 0x88); // 130 146 162 178 ...
  tb = _mm512_shuffle_f32x4((__m512)ttb, (__m512)ttf, 0x88); // 131 147 163 179 ...
  tc = _mm512_shuffle_f32x4((__m512)tt8, (__m512)ttc, 0xdd); // 132 148 164 180 ...
  td = _mm512_shuffle_f32x4((__m512)tt9, (__m512)ttd, 0xdd); // 133 149 165 181 ...
  te = _mm512_shuffle_f32x4((__m512)tta, (__m512)tte, 0xdd); // 134 150 166 182 ...
  tf = _mm512_shuffle_f32x4((__m512)ttb, (__m512)ttf, 0xdd); // 135 151 167 183 ...

  row0 = _mm512_shuffle_f32x4(t0, t8, 0x88); //   0  16  32  48  64  80  96 112 ... 240
  row1 = _mm512_shuffle_f32x4(t1, t9, 0x88); //   1  17  33  49  66  81  97 113 ... 241
  row2 = _mm512_shuffle_f32x4(t2, ta, 0x88); //   2  18  34  50  67  82  98 114 ... 242
  row3 = _mm512_shuffle_f32x4(t3, tb, 0x88); //   3  19  35  51  68  83  99 115 ... 243
  row4 = _mm512_shuffle_f32x4(t4, tc, 0x88); //   4 ...
  row5 = _mm512_shuffle_f32x4(t5, td, 0x88); //   5 ...
  row6 = _mm512_shuffle_f32x4(t6, te, 0x88); //   6 ...
  row7 = _mm512_shuffle_f32x4(t7, tf, 0x88); //   7 ...
  row8 = _mm512_shuffle_f32x4(t0, t8, 0xdd); //   8 ...
  row9 = _mm512_shuffle_f32x4(t1, t9, 0xdd); //   9 ...
  row10 = _mm512_shuffle_f32x4(t2, ta, 0xdd); //  10 ...
  row11 = _mm512_shuffle_f32x4(t3, tb, 0xdd); //  11 ...
  row12 = _mm512_shuffle_f32x4(t4, tc, 0xdd); //  12 ...
  row13 = _mm512_shuffle_f32x4(t5, td, 0xdd); //  13 ...
  row14 = _mm512_shuffle_f32x4(t6, te, 0xdd); //  14 ...
  row15 = _mm512_shuffle_f32x4(t7, tf, 0xdd); //  15  31  47  63  79  96 111 127 ... 255
}

__m512i AVX512Util::Reverse8(__m512i &v) {
  return _mm512_permutexvar_epi64(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), v);
}

__m512d AVX512Util::Reverse8(__m512d &v) {
  return _mm512_permutexvar_pd(_mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0), v);
}

__m512i AVX512Util::Reverse16(__m512i &v) {
  return _mm512_permutexvar_epi32(_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                                    7, 6, 5, 4, 3, 2, 1, 0), v);
}

__m512 AVX512Util::Reverse16(__m512 &v) {
  return _mm512_permutexvar_ps(_mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8,
                                                 7, 6, 5, 4, 3, 2, 1, 0), v);
}


void AVX512Util::IntraRegisterSort8x8(__m512i &a8, __m512i &b8) {
  __m512i mina, maxa, minb, maxb;
  // phase 1
  MinMax8(a8, b8);
  auto a8_1 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), a8);
  auto b8_1 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), b8);

  MinMax8(a8, a8_1, mina, maxa);
  MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm512_permutex2var_epi64(mina, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxa);
  auto b4 = _mm512_permutex2var_epi64(minb, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxb);

  // TODO: Convert
  // phase 2
//  auto a4_1 = _mm512_shuffle_epi32(a4, 0x4e);
//  auto b4_1 = _mm512_shuffle_epi32(b4, 0x4e);
//
//  MinMax8(a4, a4_1, mina, maxa);
//  MinMax8(b4, b4_1, minb, maxb);
//
//  auto a2 = _mm512_unpacklo_epi64(mina, maxa);
//  auto b2 = _mm512_unpacklo_epi64(minb, maxb);
//
//  // phase 3
//  auto a2_1 = _mm512_shuffle_epi32(a2, 0xb1);
//  auto b2_1 = _mm512_shuffle_epi32(b2, 0xb1);
//
//  MinMax8(a2, a2_1, mina, maxa);
//  MinMax8(b2, b2_1, minb, maxb);
//
//  a8 = _mm512_blend_epi32(mina, maxa, 0xaa);
//  b8 = _mm512_blend_epi32(minb, maxb, 0xaa);
}

void AVX512Util::IntraRegisterSort8x8(__m512d &a8, __m512d &b8) {
  __m512d mina, maxa, minb, maxb;
  // phase 1
  MinMax8(a8, b8);
  auto a8_1 = _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), a8);
  auto b8_1 = _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), b8);

  MinMax8(a8, a8_1, mina, maxa);
  MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm512_permutex2var_pd(mina, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxa);
  auto b4 = _mm512_permutex2var_pd(minb, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxb);

  // TODO: Convert
  // phase 2
//  auto a4_1 = _mm512_shuffle_ps(a4, a4, 0x4e);
//  auto b4_1 = _mm512_shuffle_ps(b4, b4, 0x4e);
//
//  MinMax8(a4, a4_1, mina, maxa);
//  MinMax8(b4, b4_1, minb, maxb);
//
//  auto a2 = _mm512_unpacklo_pd(mina, maxa);
//  auto b2 = _mm512_unpacklo_pd(minb, maxb);
//
//  // phase 3
//  auto a2_1 = _mm512_shuffle_ps(a2, a2, 0xb1);
//  auto b2_1 = _mm512_shuffle_ps(b2, b2, 0xb1);
//
//  MinMax8(a2, a2_1, mina, maxa);
//  MinMax8(b2, b2_1, minb, maxb);
//
//  a8 = _mm512_blend_ps(mina, maxa, 0xaa);
//  b8 = _mm512_blend_ps(minb, maxb, 0xaa);
}

void AVX512Util::IntraRegisterSort16x16(__m512i& a16, __m512i& b16) {
  __m512i mina, maxa, minb, maxb;
  // phase 1
  print_arr((int*)(&a16), 0, 16, "a16: ");
  print_arr((int*)(&b16), 0, 16, "b16: ");

  MinMax16(a16, b16);

  print_arr((int*)(&a16), 0, 16, "min a16: ");
  print_arr((int*)(&b16), 0, 16, "max b16: ");

  auto a16_1 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), a16);
  auto b16_1 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), b16);

  print_arr((int*)(&a16_1), 0, 16, "a16_1: ");
  print_arr((int*)(&b16_1), 0, 16, "b16_1: ");
  
  MinMax16(a16, a16_1, mina, maxa);
  MinMax16(b16, b16_1, minb, maxb);

  print_arr((int*)(&mina), 0, 16, "a16_1 -> mina: ");
  print_arr((int*)(&minb), 0, 16, "b16_1 -> minb: ");

  // phase 2
  auto a8 = _mm512_mask_blend_epi32((__mmask16)(0xff00), mina, maxa);
  auto b8 = _mm512_mask_blend_epi32((__mmask16)(0xff00), minb, maxb);

  print_arr((int*)(&a8), 0, 16, "a8: ");
  print_arr((int*)(&b8), 0, 16, "b8: ");

  auto a8_1 = _mm512_shuffle_i32x4(a8, a8, 0xb1);
  auto b8_1 = _mm512_shuffle_i32x4(b8, b8, 0xb1);

  print_arr((int*)(&a8_1), 0, 16, "a8_1: ");
  print_arr((int*)(&b8_1), 0, 16, "b8_1: ");

  MinMax16(a8, a8_1, mina, maxa);
  MinMax16(b8, b8_1, minb, maxb);

  print_arr((int*)(&mina), 0, 16, "a8_1 -> mina: ");
  print_arr((int*)(&minb), 0, 16, "b8_1 -> minb: ");

  // phase 3
  auto a4 = _mm512_mask_blend_epi32((__mmask16)(0xf0f0), mina, maxa);
  auto b4 = _mm512_mask_blend_epi32((__mmask16)(0xf0f0), minb, maxb);

  print_arr((int*)(&a4), 0, 16, "a4: ");
  print_arr((int*)(&b4), 0, 16, "b4: ");

  // https://clang.llvm.org/doxygen/avx512fintrin_8h_source.html
  auto a4_1 = _mm512_shuffle_epi32(a4, _MM_PERM_BADC); // 0x4e
  auto b4_1 = _mm512_shuffle_epi32(b4, _MM_PERM_BADC); // 0x4e

  print_arr((int*)(&a4_1), 0, 16, "a4_1: ");
  print_arr((int*)(&b4_1), 0, 16, "b4_1: ");

  MinMax16(a4, a4_1, mina, maxa);
  MinMax16(b4, b4_1, minb, maxb);

  print_arr((int*)(&mina), 0, 16, "a4_1 -> mina: ");
  print_arr((int*)(&minb), 0, 16, "b4_1 -> minb: ");

  // phase 4
  auto a2 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), mina, maxa);
  auto b2 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), minb, maxb);

  print_arr((int*)(&a2), 0, 16, "a2: ");
  print_arr((int*)(&b2), 0, 16, "b2: ");

  auto a2_1 = _mm512_shuffle_epi32(a2, _MM_PERM_CDAB); // 0xb1
  auto b2_1 = _mm512_shuffle_epi32(b2, _MM_PERM_CDAB); // 0xb1

  print_arr((int*)(&a2_1), 0, 16, "a2_1: ");
  print_arr((int*)(&b2_1), 0, 16, "b2_1: ");

  a16 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), a2, a2_1);
  b16 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), b2, b2_1);

  print_arr((int*)(&a16), 0, 16, "a16: ");
  print_arr((int*)(&b16), 0, 16, "b16: ");
}

void AVX512Util::IntraRegisterSort16x16(__m512& a16, __m512& b16) {
  __m512 mina, maxa, minb, maxb;
  // phase 1
  MinMax16(a16, b16);
  auto a16_1 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), a16);
  auto b16_1 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), b16);

  MinMax16(a16, a16_1, mina, maxa);
  MinMax16(b16, b16_1, minb, maxb);

  // phase 2
  auto a8 = _mm512_mask_blend_ps((__mmask16)(0xff00), mina, maxa);
  auto b8 = _mm512_mask_blend_ps((__mmask16)(0xff00), minb, maxb);

  auto a8_1 = _mm512_shuffle_f32x4(a8, a8, 0xb1);
  auto b8_1 = _mm512_shuffle_f32x4(b8, b8, 0xb1);

  MinMax16(a8, a8_1, mina, maxa);
  MinMax16(b8, b8_1, minb, maxb);

  // phase 3
  auto a4 = _mm512_mask_blend_ps((__mmask16)(0xf0f0), mina, maxa);
  auto b4 = _mm512_mask_blend_ps((__mmask16)(0xf0f0), minb, maxb);

  auto a4_1 = _mm512_shuffle_ps(a4, a4, 0x4e);
  auto b4_1 = _mm512_shuffle_ps(b4, b4, 0x4e);

  MinMax16(a4, a4_1, mina, maxa);
  MinMax16(b4, b4_1, minb, maxb);

  // phase 4
  auto a2 = _mm512_mask_blend_ps((__mmask16)(0xcccc), mina, maxa);
  auto b2 = _mm512_mask_blend_ps((__mmask16)(0xcccc), minb, maxb);

  auto a2_1 = _mm512_shuffle_ps(a2, a2, 0xb1);
  auto b2_1 = _mm512_shuffle_ps(b2, b2, 0xb1);

  a16 = _mm512_mask_blend_ps((__mmask16)(0xcccc), a2, a2_1);
  b16 = _mm512_mask_blend_ps((__mmask16)(0xcccc), b2, b2_1);
}

void AVX512Util::BitonicMerge8(__m512i& a, __m512i& b) {
  
}

void AVX512Util::BitonicMerge8(__m512d& a, __m512d& b) {
  
}

void AVX512Util::BitonicMerge16(__m512i& a, __m512i& b) {
  b = Reverse16(b);
  IntraRegisterSort16x16(a,b);
}

void AVX512Util::BitonicMerge16(__m512& a, __m512& b) {
  b = Reverse16(b);
  IntraRegisterSort16x16(a,b);
}


#endif