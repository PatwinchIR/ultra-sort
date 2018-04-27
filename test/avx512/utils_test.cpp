#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"

TEST(UtilsTest, AVX512LoadStore32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 16);
  TestUtil::PopulateSeqArray(b, 16, 32);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::StoreReg(ra, b);
  AVX512Util::StoreReg(rb, a);
  for(int i = 0; i < 16; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 16);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512LoadStore64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 4);
  aligned_init<int64_t>(b, 4);
  TestUtil::PopulateSeqArray<int64_t>(a, 0, 8);
  TestUtil::PopulateSeqArray<int64_t>(b, 8, 16);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::StoreReg(ra, b);
  AVX512Util::StoreReg(rb, a);
  for(int i = 0; i < 8; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 8);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax16Int32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::RandGenInt(a, 16, -10, 10);
  TestUtil::RandGenInt(b, 16, -10, 10);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 16; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax16Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  float lo = -10;
  float hi = 10;
  TestUtil::RandGenFloat(a, 16, lo, hi);
  TestUtil::RandGenFloat(b, 16, lo, hi);
  __m512 ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 16; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax8Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);
  TestUtil::RandGenInt<int64_t>(a, 8, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 8, -10, 10);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax8Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 8);
  aligned_init<double>(b, 8);
  TestUtil::RandGenFloat<double>(a, 8, -10, 10);
  TestUtil::RandGenFloat<double>(b, 8, -10, 10);
  __m512d ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicSort16x16Int32BitTest) {
  int *arr;
  aligned_init(arr, 256);
  TestUtil::RandGenInt(arr, 256, -10, 10);
  __m512i r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512Util::BitonicSort16x16(r[0], r[1], r[2], r[3],
                               r[4], r[5], r[6], r[7],
                               r[8], r[9], r[10], r[11],
                               r[12], r[13], r[14], r[15]);
  for(int i = 0; i < 16; i++) {
    AVX512Util::StoreReg(r[i], arr + i*16);
  }
  for(int i = 16; i < 256; i+=16) {
    for(int j = i; j < i + 16; j++) {
      EXPECT_LE(arr[j-16], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicSort16x16Float32BitTest) {
  float *arr;
  aligned_init(arr, 256);
  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);
  __m512 r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512Util::BitonicSort16x16(r[0], r[1], r[2], r[3],
                               r[4], r[5], r[6], r[7],
                               r[8], r[9], r[10], r[11],
                               r[12], r[13], r[14], r[15]);
  for(int i = 0; i < 16; i++) {
    AVX512Util::StoreReg(r[i], arr + i*16);
  }
  for(int i = 16; i < 256; i+=16) {
    for(int j = i; j < i + 16; j++) {
      EXPECT_LE(arr[j-16], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicSort8x8Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 64);
  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);
  __m512i r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512Util::BitonicSort8x8(r[0], r[1], r[2], r[3],
                             r[4], r[5], r[6], r[7]);
  for(int i = 0; i < 8; i++) {
    AVX512Util::StoreReg(r[i], arr + i*8);
  }
  for(int i = 8; i < 64; i+=8) {
    for(int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicSort8x8Float64BitTest) {
  double *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);
  __m512d r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512Util::BitonicSort8x8(r[0], r[1], r[2], r[3],
                             r[4], r[5], r[6], r[7]);
  for(int i = 0; i < 8; i++) {
    AVX512Util::StoreReg(r[i], arr + i*8);
  }
  for(int i = 8; i < 64; i+=8) {
    for(int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicMerge16Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 32, 2);
  TestUtil::PopulateSeqArray(b, 1, 32, 2);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  int ab[32];
  for(int i = 0; i < 32; i++) {
    if(i < 16) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 16];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge16Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 32, 2);
  TestUtil::PopulateSeqArray(b, 1, 32, 2);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  int ab[32];
  for(int i = 0; i < 32; i++) {
    if(i < 16) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 16];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge8Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);
  TestUtil::PopulateSeqArray<int64_t>(a, 0, 8, 2);
  TestUtil::PopulateSeqArray<int64_t>(b, 1, 8, 2);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  int64_t ab[8];
  for(int i = 0; i < 8; i++) {
    if(i < 4) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 4];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge8Float64BitTest) {
  double *a;
  double *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 8, 2);
  TestUtil::PopulateSeqArray(b, 1, 8, 2);
  __m512d ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  double ab[8];
  for(int i = 0; i < 8; i++) {
    if(i < 4) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 4];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}
