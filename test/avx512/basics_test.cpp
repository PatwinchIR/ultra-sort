#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"
#include "common.h"

TEST(BasicsTest, AVX512MaxDoubleTest) {
  double *c;
  aligned_init<double>(c, 8);
  __m512d ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  ra = _mm512_setr_pd(3, 1, 5, 7, 11, 13, 9, 15);
  rb = _mm512_setr_pd(5, 4, 2, 9, 10, 14, 11, 16);
  rc = _mm512_max_pd(ra, rb);
  AVX512Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 5);
  EXPECT_DOUBLE_EQ(c[1], 4);
  EXPECT_DOUBLE_EQ(c[2], 5);
  EXPECT_DOUBLE_EQ(c[3], 9);
  EXPECT_DOUBLE_EQ(c[4], 11);
  EXPECT_DOUBLE_EQ(c[5], 14);
  EXPECT_DOUBLE_EQ(c[6], 11);
  EXPECT_DOUBLE_EQ(c[7], 16);
  delete c;
}

TEST(BasicsTest, AVX512UnpackLoHiTest) {
  float *c;
  aligned_init<float>(c, 16);
  __m512 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi
  ra = _mm512_setr_ps(3, 1, 5, 7, 8, 12, 6, 2, 5, 4, 2, 9, 1, 3, 4, 5);
  rb = _mm512_setr_ps(5, 4, 2, 9, 1, 3, 4, 5, 3, 1, 5, 7, 8, 12, 6, 2);
  // picks out lower order bits
  rc = _mm512_unpacklo_ps(ra, rb);
  AVX512Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 3);
  EXPECT_DOUBLE_EQ(c[1], 5);
  EXPECT_DOUBLE_EQ(c[2], 1);
  EXPECT_DOUBLE_EQ(c[3], 4);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 1);
  EXPECT_DOUBLE_EQ(c[6], 12);
  EXPECT_DOUBLE_EQ(c[7], 3);
  EXPECT_DOUBLE_EQ(c[8], 5);
  EXPECT_DOUBLE_EQ(c[9], 3);
  EXPECT_DOUBLE_EQ(c[10], 4);
  EXPECT_DOUBLE_EQ(c[11], 1);
  EXPECT_DOUBLE_EQ(c[12], 1);
  EXPECT_DOUBLE_EQ(c[13], 8);
  EXPECT_DOUBLE_EQ(c[14], 3);
  EXPECT_DOUBLE_EQ(c[15], 12);
  // picks out higher order bits
  rc = _mm512_unpackhi_ps(ra, rb);
  AVX512Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 5);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 9);
  EXPECT_DOUBLE_EQ(c[4], 6);
  EXPECT_DOUBLE_EQ(c[5], 4);
  EXPECT_DOUBLE_EQ(c[6], 2);
  EXPECT_DOUBLE_EQ(c[7], 5);
  EXPECT_DOUBLE_EQ(c[8], 2);
  EXPECT_DOUBLE_EQ(c[9], 5);
  EXPECT_DOUBLE_EQ(c[10], 9);
  EXPECT_DOUBLE_EQ(c[11], 7);
  EXPECT_DOUBLE_EQ(c[12], 4);
  EXPECT_DOUBLE_EQ(c[13], 6);
  EXPECT_DOUBLE_EQ(c[14], 5);
  EXPECT_DOUBLE_EQ(c[15], 2);
  delete c;
}

TEST(BasicsTest, AVX512ShuffleTest) {
  float *c;
  aligned_init<float>(c, 16);
  __m512 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  ra = _mm512_setr_ps(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  rb = _mm512_setr_ps(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  rc = _mm512_shuffle_ps(ra, rb, 0b01110100);
  AVX512Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 3);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 10);
  EXPECT_DOUBLE_EQ(c[6], 15);
  EXPECT_DOUBLE_EQ(c[7], 11);
  EXPECT_DOUBLE_EQ(c[8], 16);
  EXPECT_DOUBLE_EQ(c[9], 18);
  EXPECT_DOUBLE_EQ(c[10], 23);
  EXPECT_DOUBLE_EQ(c[11], 19);
  EXPECT_DOUBLE_EQ(c[12], 24);
  EXPECT_DOUBLE_EQ(c[13], 26);
  EXPECT_DOUBLE_EQ(c[14], 31);
  EXPECT_DOUBLE_EQ(c[15], 27);
  rc = _mm512_shuffle_ps(ra, rb, 0b10110000);
  // 00 00 11 10
  AVX512Util::StoreReg(rc, c);
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 0);
  EXPECT_DOUBLE_EQ(c[2], 7);
  EXPECT_DOUBLE_EQ(c[3], 5);
  EXPECT_DOUBLE_EQ(c[4], 8);
  EXPECT_DOUBLE_EQ(c[5], 8);
  EXPECT_DOUBLE_EQ(c[6], 15);
  EXPECT_DOUBLE_EQ(c[7], 13);
  EXPECT_DOUBLE_EQ(c[8], 16);
  EXPECT_DOUBLE_EQ(c[9], 16);
  EXPECT_DOUBLE_EQ(c[10], 23);
  EXPECT_DOUBLE_EQ(c[11], 21);
  EXPECT_DOUBLE_EQ(c[12], 24);
  EXPECT_DOUBLE_EQ(c[13], 24);
  EXPECT_DOUBLE_EQ(c[14], 31);
  EXPECT_DOUBLE_EQ(c[15], 29);
  // Alternate convenience fn for representing control bits
  EXPECT_EQ(_MM_SHUFFLE(2, 3, 0, 0), 0b10110000);
  delete c;
}

TEST(BasicsTest, AVX512BlendTest) {
  float *c;
  aligned_init<float>(c, 16);
  __m512 ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  ra = _mm512_setr_ps(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  rb = _mm512_setr_ps(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  rc = _mm512_blend_ps(ra, rb, 0b01110100);
  AVX512Util::StoreReg(rc, c);
  // 00101110
  EXPECT_DOUBLE_EQ(c[0], 0);
  EXPECT_DOUBLE_EQ(c[1], 2);
  EXPECT_DOUBLE_EQ(c[2], 5);
  EXPECT_DOUBLE_EQ(c[3], 6);
  EXPECT_DOUBLE_EQ(c[4], 9);
  EXPECT_DOUBLE_EQ(c[5], 11);
  EXPECT_DOUBLE_EQ(c[6], 13);
  EXPECT_DOUBLE_EQ(c[7], 14);
  EXPECT_DOUBLE_EQ(c[8], 16);
  EXPECT_DOUBLE_EQ(c[9], 18);
  EXPECT_DOUBLE_EQ(c[10], 21);
  EXPECT_DOUBLE_EQ(c[11], 22);
  EXPECT_DOUBLE_EQ(c[12], 25);
  EXPECT_DOUBLE_EQ(c[13], 27);
  EXPECT_DOUBLE_EQ(c[14], 29);
  EXPECT_DOUBLE_EQ(c[15], 30);
  delete c;
}

TEST(BasicsTest, AVX512TestBed) {
  int *c;
  aligned_init<int>(c, 16);
  __m512i ra, rb, rc;
  // need to set in reverse order since Intel's Little Endian arch
  // low ---------> hi           |
  ra = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
  rb = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
  rc = _mm512_shuffle_epi32(ra, 0x4e);
  // 01 00 11 10 => 10 11 00 01
  // 4, 6, 0, 2, 12, 14, 8, 10
  AVX512Util::StoreReg(rc, c);
//  print_arr<int>(c, 0, 8, "rc: ");
  delete(c);
}

