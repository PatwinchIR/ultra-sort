#include "avx512/sort_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"

#ifdef AVX512
namespace avx512 {
TEST(SortUtilTest, AVX512SortBlock256Int32BitTest) {
  int *arr;
  aligned_init<int>(arr, 256);
  TestUtil::RandGenInt<int>(arr, 256, -10, 10);

  auto check_arr = (int *) malloc(256 * sizeof(int));
  auto temp_arr = (int *) malloc(16 * sizeof(int));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  SortBlock256<int, __m512i>(arr, 0);

  for (int i = 0; i < 256; i++) {
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

  auto *check_arr = (float *) malloc(256 * sizeof(float));
  auto *temp_arr = (float *) malloc(16 * sizeof(float));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  SortBlock256<float, __m512>(arr, 0);

  for (int i = 0; i < 256; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}

TEST(SortUtilTest, AVX512SortBlock64Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 64);
  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);

  auto check_arr = (int64_t *) malloc(64 * sizeof(int64_t));
  auto temp_arr = (int64_t *) malloc(8 * sizeof(int64_t));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  SortBlock64<int64_t, __m512i>(arr, 0);

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}

TEST(SortUtilTest, AVX512SortBlock64Float64BitTest) {
  double *arr;
  aligned_init<double>(arr, 64);
  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);

  auto check_arr = (double *) malloc(64 * sizeof(double));
  auto temp_arr = (double *) malloc(8 * sizeof(double));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  SortBlock64<double, __m512d>(arr, 0);

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}

}
#endif