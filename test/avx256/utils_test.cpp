#include <avx256/sort_util.h>
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

TEST(UtilsTest, AVX256LoadStore32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 8);
  TestUtil::PopulateSeqArray(b, 8, 16);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::StoreReg(ra, b);
  AVX256Util::StoreReg(rb, a);
  for(int i = 0; i < 8; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 8);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256LoadStore64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 4);
  aligned_init<int64_t>(b, 4);
  TestUtil::PopulateSeqArray<int64_t>(a, 0, 4);
  TestUtil::PopulateSeqArray<int64_t>(b, 4, 8);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::StoreReg(ra, b);
  AVX256Util::StoreReg(rb, a);
  for(int i = 0; i < 4; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 4);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256MinMax8Int32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::RandGenInt(a, 8, -10, 10);
  TestUtil::RandGenInt(b, 8, -10, 10);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::MinMax8(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256MinMax8Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  float lo = -10;
  float hi = 10;
  TestUtil::RandGenFloat(a, 8, lo, hi);
  TestUtil::RandGenFloat(b, 8, lo, hi);
  __m256 ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::MinMax8(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256MinMax4Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 4);
  aligned_init<int64_t>(b, 4);
  TestUtil::RandGenInt<int64_t>(a, 4, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 4, -10, 10);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::MinMax4(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  for(int i = 0; i < 4; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256MaskedMinMax8Int32Test) {
  // For masked tests, we assume k,v follow one another with native type
  // i.e. for 32-bit int k,v array -> Assume its flattened out to an int array
  // 4 32-bit key-value integers will fit in AVX2 register
  // This test is currently weak - only checks for sorted order
  using T = int;
  int unit_size = 4;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenIntRecords(kv_flat1, unit_size, -10, 10, 0);
  TestUtil::RandGenIntRecords(kv_flat2, unit_size, -10, 10, unit_size*2);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T,T>(kv_flat1[2*i + 1], kv_flat1[2*i]));
    kv_map.insert(std::pair<T,T>(kv_flat2[2*i + 1], kv_flat2[2*i]));
  }

  __m256i r1, r2;
  AVX256Util::LoadReg(r1, kv_flat1);
  AVX256Util::LoadReg(r2, kv_flat2);
  AVX256Util::MaskedMinMax8(r1, r2);
  AVX256Util::StoreReg(r1, kv_flat1);
  AVX256Util::StoreReg(r2, kv_flat2);
  for(int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2*i], kv_flat2[2*i]);
    EXPECT_EQ(kv_flat1[2*i], kv_map[kv_flat1[2*i + 1]]);
    EXPECT_EQ(kv_flat2[2*i], kv_map[kv_flat2[2*i + 1]]);
  }
  delete[](kv_flat1);
  delete[](kv_flat2);
}

TEST(UtilsTest, AVX256MaskedMinMax8FloatTest) {
  // For masked tests, we assume k,v follow one another with native type
  // i.e. for 32-bit int k,v array -> Assume its flattened out to an int array
  // 4 32-bit key-value integers will fit in AVX2 register
  // This test is currently weak - only checks for sorted order
  using T = float;
  int unit_size = 4;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenFloatRecords(kv_flat1, unit_size, -10.0f, 10.0f, 0);
  TestUtil::RandGenFloatRecords(kv_flat2, unit_size, -10.0f, 10.0f, unit_size*2);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T,T>(kv_flat1[2*i + 1], kv_flat1[2*i]));
    kv_map.insert(std::pair<T,T>(kv_flat2[2*i + 1], kv_flat2[2*i]));
  }

  __m256 r1, r2;
  AVX256Util::LoadReg(r1, kv_flat1);
  AVX256Util::LoadReg(r2, kv_flat2);
  AVX256Util::MaskedMinMax8(r1, r2);
  AVX256Util::StoreReg(r1, kv_flat1);
  AVX256Util::StoreReg(r2, kv_flat2);
  for(int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2*i], kv_flat2[2*i]);
    EXPECT_EQ(kv_flat1[2*i], kv_map[kv_flat1[2*i + 1]]);
    EXPECT_EQ(kv_flat2[2*i], kv_map[kv_flat2[2*i + 1]]);
  }
  delete[](kv_flat1);
  delete[](kv_flat2);
}

//TEST(UtilsTest, AVX256MaskedMinMax4Int64Test) {
//  // For masked tests, we assume k,v follow one another with native type
//  // i.e. for 32-bit int k,v array -> Assume its flattened out to an int array
//  // 4 32-bit key-value integers will fit in AVX2 register
//  // This test is currently weak - only checks for sorted order
//  using T = int64_t;
//  int unit_size = 2;
//  T *kv_flat1, *kv_flat2;
//  TestUtil::RandGenIntRecords(kv_flat1, unit_size*2, -10ll, 10ll, 0);
//  TestUtil::RandGenIntRecords(kv_flat2, unit_size*2, -10ll, 10ll, unit_size*2);
//  std::map<T,T> kv_map;
//  for (int i = 0; i < unit_size; ++i) {
//    kv_map.insert(std::pair<T,T>(kv_flat1[2*i + 1], kv_flat1[2*i]));
//    kv_map.insert(std::pair<T,T>(kv_flat2[2*i + 1], kv_flat2[2*i]));
//  }
//
//  __m256i r1, r2;
//  AVX256Util::LoadReg(r1, kv_flat1);
//  AVX256Util::LoadReg(r2, kv_flat2);
//  AVX256Util::MaskedMinMax4(r1, r2);
//  AVX256Util::StoreReg(r1, kv_flat1);
//  AVX256Util::StoreReg(r2, kv_flat2);
//  for(int i = 0; i < unit_size; i++) {
//    EXPECT_LE(kv_flat1[2*i], kv_flat2[2*i]);
//    EXPECT_EQ(kv_flat1[2*i], kv_map[kv_flat1[2*i + 1]]);
//    EXPECT_EQ(kv_flat2[2*i], kv_map[kv_flat2[2*i + 1]]);
//  }
//  delete[](kv_flat1);
//  delete[](kv_flat2);
//}

//TEST(UtilsTest, AVX256MaskedMinMax4Int64Test) {
//  double *a;
//  double *b;
//  aligned_init<double>(a, 4);
//  aligned_init<double>(b, 4);
//  TestUtil::RandGenFloat<double>(a, 4, -10, 10);
//  TestUtil::RandGenFloat<double>(b, 4, -10, 10);
//  __m256d ra, rb;
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256Util::MinMax4(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//  for(int i = 0; i < 4; i++) {
//    EXPECT_LE(a[i], b[i]);
//  }
//  delete[](a);
//  delete[](b);
//}
//
