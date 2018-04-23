#include "avx256/utils.h"
#include "common.h"

#ifdef AVX2

/**
 * Load and Store Instructions
 */

template <typename InType, typename RegType>
void AVX256Util::LoadReg(RegType &r, InType *arr) {
  r = *((RegType*)arr);
}

template void AVX256Util::LoadReg<int, __m256i>( __m256i &r, int *arr);
template void AVX256Util::LoadReg<int64_t, __m256i>(__m256i &r, int64_t *arr);
template void AVX256Util::LoadReg<float, __m256>(__m256 &r, float *arr);
template void AVX256Util::LoadReg<double, __m256d>(__m256d &r, double *arr);

template <typename InType, typename RegType>
void AVX256Util::StoreReg(const RegType &r, InType *arr) {
  *((RegType*)arr) = r;
}

template void AVX256Util::StoreReg<int, __m256i>(const __m256i &r, int *arr);
template void AVX256Util::StoreReg<int64_t, __m256i>(const __m256i &r, int64_t *arr);
template void AVX256Util::StoreReg<float, __m256>(const __m256 &r, float *arr);
template void AVX256Util::StoreReg<double, __m256d>(const __m256d &r, double *arr);

/**
 * Converter Utilities
 * Int => Double
 * Double => Int
 */

__m256d AVX256Util::Int64ToDoubleReg(const __m256i &repi64) {
  int64_t temp[4];
  StoreReg(repi64, temp);
  return _mm256_setr_pd(temp[0], temp[1], temp[2], temp[3]);
}

__m256i AVX256Util::DoubleToInt64Reg(const __m256d &rd) {
  double temp[4];
  StoreReg(rd, temp);
  return _mm256_setr_epi64x((int64_t)temp[0], (int64_t)temp[1], (int64_t)temp[2], (int64_t)temp[3]);
}

/**
 * MinMax functions
 */

void AVX256Util::MinMax8(__m256i &a, __m256i &b) {
  __m256i c = a;
  a = _mm256_min_epi32(a, b);
  b = _mm256_max_epi32(c, b);
}

void AVX256Util::MinMax8(const __m256i &a, const __m256i &b, __m256i &minab, __m256i &maxab) {
  minab = _mm256_min_epi32(a, b);
  maxab = _mm256_max_epi32(a, b);
}

void AVX256Util::MinMax8(__m256 &a, __m256 &b) {
  __m256 c = a;
  a = _mm256_min_ps(a, b);
  b = _mm256_max_ps(c, b);
}

void AVX256Util::MinMax8(const __m256 &a, const __m256 &b, __m256 &minab, __m256 &maxab) {
  minab = _mm256_min_ps(a, b);
  maxab = _mm256_max_ps(a, b);
}

void AVX256Util::MinMax4(__m256i &a, __m256i &b) {
  __m256d a_d = Int64ToDoubleReg(a);
  __m256d b_d = Int64ToDoubleReg(b);
  __m256d  c_d = a_d;
  a_d = _mm256_min_pd(a_d, b_d);
  b_d = _mm256_max_pd(c_d, b_d);
  a = DoubleToInt64Reg(a_d);
  b = DoubleToInt64Reg(b_d);
}

void AVX256Util::MinMax4(__m256d &a, __m256d &b) {
  __m256d c = a;
  a = _mm256_min_pd(a, b);
  b = _mm256_max_pd(c, b);
}

/**
 * Bitonic Sorting networks:
 * 8x8 networks: int32, float32
 * 4x4 networks: int64
 */
template <typename T>
void AVX256Util::BitonicSort8x8(T &r0,
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

// 32 bit ints, floats
template void AVX256Util::BitonicSort8x8<__m256i>(__m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&, __m256i&);
template void AVX256Util::BitonicSort8x8<__m256>(__m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&, __m256&);

template <typename T>
void AVX256Util::BitonicSort4x4(T &r0,
                                T &r1,
                                T &r2,
                                T &r3) {
  MinMax4(r0, r1);
  MinMax4(r2, r3);
  MinMax4(r0, r2);
  MinMax4(r1, r3);
  MinMax4(r1, r2);
}

// 64 bit ints, floats
template void AVX256Util::BitonicSort4x4<__m256i>(__m256i&, __m256i&, __m256i&, __m256i&);
template void AVX256Util::BitonicSort4x4<__m256d>(__m256d&, __m256d&, __m256d&, __m256d&);

/**
 * Bitonic Transpose:
 * 8x8 networks: int32, float32
 * 4x4 networks: int64
 */

void AVX256Util::Transpose8x8(__m256i &row0,
                              __m256i &row1,
                              __m256i &row2,
                              __m256i &row3,
                              __m256i &row4,
                              __m256i &row5,
                              __m256i &row6,
                              __m256i &row7) {
  __m256 __t0, __t1, __t2, __t3, __t4, __t5, __t6, __t7;
  __m256 __tt0, __tt1, __tt2, __tt3, __tt4, __tt5, __tt6, __tt7;
  __t0 = _mm256_unpacklo_ps((__m256)row0, (__m256)row1);
  __t1 = _mm256_unpackhi_ps((__m256)row0, (__m256)row1);
  __t2 = _mm256_unpacklo_ps((__m256)row2, (__m256)row3);
  __t3 = _mm256_unpackhi_ps((__m256)row2, (__m256)row3);
  __t4 = _mm256_unpacklo_ps((__m256)row4, (__m256)row5);
  __t5 = _mm256_unpackhi_ps((__m256)row4, (__m256)row5);
  __t6 = _mm256_unpacklo_ps((__m256)row6, (__m256)row7);
  __t7 = _mm256_unpackhi_ps((__m256)row6, (__m256)row7);
  // Note: https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
  // As provided in above link, it is relatively safe/free to run float shuffles on integers(not other way around)
  __tt0 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(1,0,1,0));
  __tt1 = _mm256_shuffle_ps(__t0,__t2,_MM_SHUFFLE(3,2,3,2));
  __tt2 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(1,0,1,0));
  __tt3 = _mm256_shuffle_ps(__t1,__t3,_MM_SHUFFLE(3,2,3,2));
  __tt4 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(1,0,1,0));
  __tt5 = _mm256_shuffle_ps(__t4,__t6,_MM_SHUFFLE(3,2,3,2));
  __tt6 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(1,0,1,0));
  __tt7 = _mm256_shuffle_ps(__t5,__t7,_MM_SHUFFLE(3,2,3,2));
  row0 = (__m256i)_mm256_permute2f128_ps(__tt0, __tt4, 0x20);
  row1 = (__m256i)_mm256_permute2f128_ps(__tt1, __tt5, 0x20);
  row2 = (__m256i)_mm256_permute2f128_ps(__tt2, __tt6, 0x20);
  row3 = (__m256i)_mm256_permute2f128_ps(__tt3, __tt7, 0x20);
  row4 = (__m256i)_mm256_permute2f128_ps(__tt0, __tt4, 0x31);
  row5 = (__m256i)_mm256_permute2f128_ps(__tt1, __tt5, 0x31);
  row6 = (__m256i)_mm256_permute2f128_ps(__tt2, __tt6, 0x31);
  row7 = (__m256i)_mm256_permute2f128_ps(__tt3, __tt7, 0x31);
}

void AVX256Util::Transpose8x8(__m256 &row0,
                              __m256 &row1,
                              __m256 &row2,
                              __m256 &row3,
                              __m256 &row4,
                              __m256 &row5,
                              __m256 &row6,
                              __m256 &row7) {
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
  // Note: https://stackoverflow.com/questions/26983569/implications-of-using-mm-shuffle-ps-on-integer-vector
  // As provided in above link, it is relatively safe/free to run float shuffles on integers(not other way around)
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

void AVX256Util::Transpose4x4(__m256i &row0,
                              __m256i &row1,
                              __m256i &row2,
                              __m256i &row3) {
  __m256i __t0, __t1, __t2, __t3;
  __t0 = _mm256_unpacklo_epi64(row0, row1);
  __t1 = _mm256_unpackhi_epi64(row0, row1);
  __t2 = _mm256_unpacklo_epi64(row2, row3);
  __t3 = _mm256_unpackhi_epi64(row2, row3);
  row0 = (__m256i)_mm256_permute2f128_ps((__m256)__t0, (__m256)__t2, 0x20);
  row1 = (__m256i)_mm256_permute2f128_ps((__m256)__t1, (__m256)__t3, 0x20);
  row2 = (__m256i)_mm256_permute2f128_ps((__m256)__t0, (__m256)__t2, 0x31);
  row3 = (__m256i)_mm256_permute2f128_ps((__m256)__t1, (__m256)__t3, 0x31);
}

__m256i AVX256Util::Reverse8(__m256i &v) {
  return _mm256_permutevar8x32_epi32(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

__m256 AVX256Util::Reverse8(__m256 &v) {
  return _mm256_permutevar8x32_ps(v, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
}

__m256i AVX256Util::Reverse4(__m256i &v) {
  return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(0, 1, 2, 3));
}

__m256d AVX256Util::Reverse4(__m256d &v) {
  return _mm256_permute4x64_pd(v, _MM_SHUFFLE(0, 1, 2, 3));
}

void AVX256Util::IntraRegisterSort8x8(__m256i &a8, __m256i &b8) {
  __m256i mina, maxa, minb, maxb;
  // phase 1
  MinMax8(a8, b8);
  auto a8_1 = _mm256_permutevar8x32_epi32(a8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
  auto b8_1 = _mm256_permutevar8x32_epi32(b8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));

  MinMax8(a8, a8_1, mina, maxa);
  MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm256_blend_epi32(mina, maxa, 0xf0);
  auto b4 = _mm256_blend_epi32(minb, maxb, 0xf0);

  // phase 2
  auto a4_1 = _mm256_shuffle_epi32(a4, 0x4e);
  auto b4_1 = _mm256_shuffle_epi32(b4, 0x4e);

  MinMax8(a4, a4_1, mina, maxa);
  MinMax8(b4, b4_1, minb, maxb);

  auto a2 = _mm256_unpacklo_epi64(mina, maxa);
  auto b2 = _mm256_unpacklo_epi64(minb, maxb);

  // phase 3
  auto a2_1 = _mm256_shuffle_epi32(a2, 0xb1);
  auto b2_1 = _mm256_shuffle_epi32(b2, 0xb1);

  MinMax8(a2, a2_1, mina, maxa);
  MinMax8(b2, b2_1, minb, maxb);

  a8 = _mm256_blend_epi32(mina, maxa, 0xaa);
  b8 = _mm256_blend_epi32(minb, maxb, 0xaa);
}

void AVX256Util::IntraRegisterSort8x8(__m256 &a8, __m256 &b8) {
  __m256 mina, maxa, minb, maxb;
  // phase 1
  MinMax8(a8, b8);
  auto a8_1 = _mm256_permutevar8x32_ps(a8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
  auto b8_1 = _mm256_permutevar8x32_ps(b8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));

  MinMax8(a8, a8_1, mina, maxa);
  MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm256_blend_ps(mina, maxa, 0xf0);
  auto b4 = _mm256_blend_ps(minb, maxb, 0xf0);

  // phase 2
  auto a4_1 = _mm256_shuffle_ps(a4, a4, 0x4e);
  auto b4_1 = _mm256_shuffle_ps(b4, b4, 0x4e);

  MinMax8(a4, a4_1, mina, maxa);
  MinMax8(b4, b4_1, minb, maxb);

  auto a2 = _mm256_unpacklo_pd(mina, maxa);
  auto b2 = _mm256_unpacklo_pd(minb, maxb);

  // phase 3
  auto a2_1 = _mm256_shuffle_ps(a2, a2, 0xb1);
  auto b2_1 = _mm256_shuffle_ps(b2, b2, 0xb1);

  MinMax8(a2, a2_1, mina, maxa);
  MinMax8(b2, b2_1, minb, maxb);

  a8 = _mm256_blend_ps(mina, maxa, 0xaa);
  b8 = _mm256_blend_ps(minb, maxb, 0xaa);
}

void AVX256Util::IntraRegisterSort4x4(__m256i &a4, __m256i &b4) {
  // Level 1
  MinMax4(a4, b4);
  auto l1p = (__m256i)_mm256_permute2f128_pd((__m256d)a4, (__m256d)b4, 0x31);
  auto h1p = (__m256i)_mm256_permute2f128_pd((__m256d)a4, (__m256d)b4, 0x20);

  // Level 2
  MinMax4(l1p, h1p);
  auto l2p = (__m256i)_mm256_shuffle_pd((__m256d)l1p, (__m256d)h1p, 0x0);
  auto h2p = (__m256i)_mm256_shuffle_pd((__m256d)l1p, (__m256d)h1p, 0xF);

  // Level 3
  MinMax4(l2p, h2p);
  auto l3p = (__m256i)_mm256_unpacklo_pd((__m256d)l2p, (__m256d)h2p);
  auto h3p = (__m256i)_mm256_unpackhi_pd((__m256d)l2p, (__m256d)h2p);

  // Finally
  a4 = (__m256i)_mm256_permute2f128_pd((__m256d)l3p, (__m256d)h3p, 0x20);
  b4 = (__m256i)_mm256_permute2f128_pd((__m256d)l3p, (__m256d)h3p, 0x31);

}

void AVX256Util::BitonicMerge8(__m256i &a, __m256i &b) {
  b = Reverse8(b);
  IntraRegisterSort8x8(a,b);
}

void AVX256Util::BitonicMerge8(__m256 &a, __m256 &b) {
  b = Reverse8(b);
  IntraRegisterSort8x8(a,b);
}

void AVX256Util::BitonicMerge4(__m256i &a, __m256i &b) {
  b = Reverse4(b);
  IntraRegisterSort4x4(a,b);
}

#endif
