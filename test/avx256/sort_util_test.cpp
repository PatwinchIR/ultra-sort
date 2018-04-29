#include "avx256/sort_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

TEST(SortUtilTest, AVX256BitonicSort8x8Int32BitTest) {
  int *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenInt(arr, 64, -10, 10);
  __m256i r[8];
  for(int i = 0; i < 8; i++) {
    AVX256Util::LoadReg(r[i], arr + i*8);
  }
  AVX256SortUtil::BitonicSort8x8(r[0], r[1], r[2], r[3],
                             r[4], r[5], r[6], r[7]);
  for(int i = 0; i < 8; i++) {
    AVX256Util::StoreReg(r[i], arr + i*8);
  }
  for(int i = 8; i < 64; i+=8) {
    for(int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(SortUtilTest, AVX256BitonicSort8x8Float32BitTest) {
  float *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);
  __m256 r[8];
  for(int i = 0; i < 8; i++) {
    AVX256Util::LoadReg(r[i], arr + i*8);
  }
  AVX256SortUtil::BitonicSort8x8(r[0], r[1], r[2], r[3],
                             r[4], r[5], r[6], r[7]);
  for(int i = 0; i < 8; i++) {
    AVX256Util::StoreReg(r[i], arr + i*8);
  }
  for(int i = 8; i < 64; i+=8) {
    for(int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(SortUtilTest, AVX256BitonicSort4x4Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 16);
  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);
  __m256i r[4];
  for(int i = 0; i < 4; i++) {
    AVX256Util::LoadReg(r[i], arr + i*4);
  }
  AVX256SortUtil::BitonicSort4x4(r[0], r[1], r[2], r[3]);
  for(int i = 0; i < 4; i++) {
    AVX256Util::StoreReg(r[i], arr + i*4);
  }
  for(int i = 4; i < 16; i+=4) {
    for(int j = i; j < i + 4; j++) {
      EXPECT_LE(arr[j-4], arr[j]);
    }
  }
  delete[](arr);
}

TEST(SortUtilTest, AVX256BitonicSort4x4Float64BitTest) {
  double *arr;
  aligned_init(arr, 16);
  TestUtil::RandGenFloat<double>(arr, 16, -10, 10);
  __m256d r[4];
  for(int i = 0; i < 4; i++) {
    AVX256Util::LoadReg(r[i], arr + i*4);
  }
  AVX256SortUtil::BitonicSort4x4(r[0], r[1], r[2], r[3]);
  for(int i = 0; i < 4; i++) {
    AVX256Util::StoreReg(r[i], arr + i*4);
  }
  for(int i = 4; i < 16; i+=4) {
    for(int j = i; j < i + 4; j++) {
      EXPECT_LE(arr[j-4], arr[j]);
    }
  }
  delete[](arr);
}