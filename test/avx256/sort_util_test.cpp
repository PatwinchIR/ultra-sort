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

TEST(SortUtilTest, AVX256MaskedBitonicSort8x8Int32BitTest) {
  using T = int;
  T *arr;
  int unit_size = 4;
  int num_rows = unit_size*2;
  TestUtil::RandGenIntRecords(arr, unit_size*num_rows, -10, 10);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size*num_rows; ++i) {
    kv_map.insert(std::pair<T,T>(arr[2*i + 1], arr[2*i]));
  }

  __m256i r[8];
  for(int i = 0; i < num_rows; i++) {
    AVX256Util::LoadReg(r[i], arr + i*unit_size*2);
  }
  AVX256SortUtil::MaskedBitonicSort8x8(r[0], r[1], r[2], r[3],
                                       r[4], r[5], r[6], r[7]);
  for(int i = 0; i < num_rows; i++) {
    AVX256Util::StoreReg(r[i], arr + i*unit_size*2);
  }

  for(int i = num_rows; i < num_rows*unit_size*2; i+=unit_size*2) {
    for(int j = i; j < i + unit_size*2; j+=2) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }

  for (int i = 0; i < unit_size*num_rows; ++i) {
    EXPECT_EQ(kv_map[arr[2*i + 1]], arr[2*i]);
  }
  delete[](arr);
}

TEST(SortUtilTest, AVX256BitonicSort8x8Float32BitTest) {
  float *arr;
  aligned_init<float>(arr, 64);
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

TEST(SortUtilTest, AVX256MaskedBitonicSort8x8Float32BitTest) {
  using T = float;
  T *arr;
  int unit_size = 4;
  int num_rows = unit_size*2;
  TestUtil::RandGenFloatRecords(arr, unit_size*num_rows, -10.0f, 10.0f);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size*num_rows; ++i) {
    kv_map.insert(std::pair<T,T>(arr[2*i + 1], arr[2*i]));
  }
  __m256 r[8];
  for(int i = 0; i < num_rows; i++) {
    AVX256Util::LoadReg(r[i], arr + i*unit_size*2);
  }
  AVX256SortUtil::MaskedBitonicSort8x8(r[0], r[1], r[2], r[3],
                                       r[4], r[5], r[6], r[7]);
  for(int i = 0; i < num_rows; i++) {
    AVX256Util::StoreReg(r[i], arr + i*unit_size*2);
  }
  for(int i = num_rows; i < num_rows*unit_size*2; i+=unit_size*2) {
    for(int j = i; j < i + unit_size*2; j+=2) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  for (int i = 0; i < unit_size*num_rows; ++i) {
    EXPECT_EQ(kv_map[arr[2*i + 1]], arr[2*i]);
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
  aligned_init<double>(arr, 16);
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

