#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"
#include "common.h"

// Introduction at: https://www.codeproject.com/Articles/874396/Crunching-Numbers-with-AVX-and-AVX
// TODO: _mm256_permute2f128_pd

TEST(BasicsTest, AVX256MaxDoubleTest) {
  double *c;
  aligned_init<double>(c, 4);
  __m256d ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  ra = _mm256_setr_pd(3, 1, 5, 7);
  rb = _mm256_setr_pd(5, 4, 2, 9);
  rc = _mm256_max_pd(ra, rb);
  AVX256Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 5);
  EXPECT_DOUBLE_EQ(c[1], 4);
  EXPECT_DOUBLE_EQ(c[2], 5);
  EXPECT_DOUBLE_EQ(c[3], 9);
  delete c;
}

TEST(BasicsTest, AVX256UnpackLoHiTest) {
  float *c;
  aligned_init<float>(c, 8);
  __m256 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi
  ra = _mm256_setr_ps(3, 1, 5, 7, 8, 12, 6, 2);
  rb = _mm256_setr_ps(5, 4, 2, 9, 1, 3, 4, 5);
  // picks out lower order bits
  rc = _mm256_unpacklo_ps(ra, rb);
  AVX256Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 3);
  EXPECT_DOUBLE_EQ(c[1], 5);
  EXPECT_DOUBLE_EQ(c[2], 1);
  EXPECT_DOUBLE_EQ(c[3], 4);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 1);
  EXPECT_DOUBLE_EQ(c[6], 12);
  EXPECT_DOUBLE_EQ(c[7], 3);
  // picks out higher order bits
  rc = _mm256_unpackhi_ps(ra, rb);
  AVX256Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 5);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 9);
  EXPECT_DOUBLE_EQ(c[4], 6);
  EXPECT_DOUBLE_EQ(c[5], 4);
  EXPECT_DOUBLE_EQ(c[6], 2);
  EXPECT_DOUBLE_EQ(c[7], 5);
  delete c;
}

TEST(BasicsTest, AVX256ShuffleTest) {
  float *c;
  aligned_init<float>(c, 8);
  __m256 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  ra = _mm256_setr_ps(0, 2, 4, 6, 8, 10, 12, 14);
  rb = _mm256_setr_ps(1, 3, 5, 7, 9, 11, 13, 15);
  rc = _mm256_shuffle_ps(ra, rb, 0b01110100);
  AVX256Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 3);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 10);
  EXPECT_DOUBLE_EQ(c[6], 15);
  EXPECT_DOUBLE_EQ(c[7], 11);
  rc = _mm256_shuffle_ps(ra, rb, 0b10110000);
  // 00 00 11 10
  AVX256Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 0);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 5);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 8);
  EXPECT_DOUBLE_EQ(c[6], 15);
  EXPECT_DOUBLE_EQ(c[7], 13);
  // Alternate convenience fn for representing control bits
  EXPECT_EQ(_MM_SHUFFLE(2, 3, 0, 0), 0b10110000);
  delete c;
}

TEST(BasicsTest, AVX256BlendTest) {
  float *c;
  aligned_init<float>(c, 8);
  __m256 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  ra = _mm256_setr_ps(0, 2, 4, 6, 8, 10, 12, 14);
  rb = _mm256_setr_ps(1, 3, 5, 7, 9, 11, 13, 15);
  rc = _mm256_blend_ps(ra, rb, 0b01110100);
  AVX256Util::StoreReg(rc, c);
  // 00101110
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 5);
  EXPECT_DOUBLE_EQ(c[3], 6);
  EXPECT_DOUBLE_EQ(c[4], 9);
  EXPECT_DOUBLE_EQ(c[5], 11);
  EXPECT_DOUBLE_EQ(c[6], 13);
  EXPECT_DOUBLE_EQ(c[7], 14);
  delete c;
}

TEST(BasicsTest, AVX256MinMaxKV) {
  // Register-type input
  __m256i ra, rb;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  // (K, V) pairs, first num is K, then V. We try to min/max by K
  ra = _mm256_setr_epi32(0, 2, 5, 8, 8, 10, 13, 15);
  rb = _mm256_setr_epi32(1, 3, 4, 10, 9, 11, 12, 14);
  // Register-type mask
  // rc:0, 0, -1, -1, 0, 0, -1, -1,
  auto rc = _mm256_permutevar8x32_epi32(_mm256_cmpgt_epi32(ra, rb), _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6));
  // 0, 0, 5, 8, 0, 0, 13, 15,
  auto rabmaxa = _mm256_and_si256(rc, ra);
  auto rabmaxb = _mm256_andnot_si256(rc, rb);
  auto rabmax = _mm256_or_si256(rabmaxa, rabmaxb);
//  rab = _mm256_andnot_si256(rab, rb);
//  print_arr((int*)(&rabmax), 0, 8, "rabmin: ");
}