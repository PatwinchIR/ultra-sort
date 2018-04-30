#include "avx512/sort_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"

#ifdef AVX512
TEST(SortUtilTest, AVX512BitonicSort16x16Int32BitTest) {
  int *arr;
  aligned_init<int>(arr, 256);
  TestUtil::RandGenInt<int>(arr, 256, -10, 10);
  __m512i r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512SortUtil::BitonicSort16x16(r[0], r[1], r[2], r[3],
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

TEST(SortUtilTest, AVX512BitonicSort16x16Float32BitTest) {
  float *arr;
  aligned_init<float>(arr, 256);
  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);
  __m512 r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512SortUtil::BitonicSort16x16(r[0], r[1], r[2], r[3],
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

TEST(SortUtilTest, AVX512BitonicSort8x8Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 64);
  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);
  __m512i r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512SortUtil::BitonicSort8x8(r[0], r[1], r[2], r[3],
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

TEST(SortUtilTest, AVX512BitonicSort8x8Float64BitTest) {
  double *arr;
  aligned_init<double>(arr, 64);
  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);
  __m512d r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512SortUtil::BitonicSort8x8(r[0], r[1], r[2], r[3],
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

TEST(SortUtilTest, AVX512SortBlock256Int32BitTest) {
  int *arr;
  aligned_init<int>(arr, 256);
  TestUtil::RandGenInt<int>(arr, 256, -10, 10);

  int *check_arr = (int *)malloc(256 * sizeof(int));
  int *temp_arr = (int *)malloc(16 * sizeof(int));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  AVX512SortUtil::SortBlock256<int, __m512i>(arr, 0);

  for(int i = 0; i < 256; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}

TEST(SortUtilTest, AVX512SortBlock256Float32BitTest) {
  float *arr;
  aligned_init<float>(arr, 256);
  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);

  auto *check_arr = (float *)malloc(256 * sizeof(float));
  auto *temp_arr = (float *)malloc(16 * sizeof(float));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  AVX512SortUtil::SortBlock256<float, __m512>(arr, 0);

  for(int i = 0; i < 256; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}
#endif