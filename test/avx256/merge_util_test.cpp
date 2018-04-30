#include <avx256/merge_util.h>
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

namespace avx2 {
TEST(MergeUtilsTest, AVX256BitonicMerge8Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 16, 2);
  TestUtil::PopulateSeqArray(b, 1, 16, 2);
  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  int ab[16];
  for (int i = 0; i < 16; i++) {
    if (i < 8) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 8];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

TEST(MergeUtilsTest, AVX256BitonicMerge8Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 16, 2);
  TestUtil::PopulateSeqArray(b, 1, 16, 2);
  __m256 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  float ab[16];
  for (int i = 0; i < 16; i++) {
    if (i < 8) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 8];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete (a);
  delete (b);
}

TEST(MergeUtilsTest, AVX256BitonicMerge4Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);
  TestUtil::PopulateSeqArray<int64_t>(a, 0, 8, 2);
  TestUtil::PopulateSeqArray<int64_t>(b, 1, 8, 2);
  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  int64_t ab[8];
  for (int i = 0; i < 8; i++) {
    if (i < 4) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 4];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

TEST(MergeUtilsTest, AVX256BitonicMerge4Float64BitTest) {
  double *a;
  double *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 8, 2);
  TestUtil::PopulateSeqArray(b, 1, 8, 2);
  __m256d ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  double ab[8];
  for (int i = 0; i < 8; i++) {
    if (i < 4) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 4];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}
}