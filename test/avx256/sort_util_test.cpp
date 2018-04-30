#include "avx256/sort_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

namespace avx2 {
TEST(SortUtilTest, AVX256SortBlock64Int32BitTest) {
  int *arr;
  aligned_init<int>(arr, 64);
  TestUtil::RandGenInt<int>(arr, 64, -10, 10);

  int *check_arr = (int *) malloc(64 * sizeof(int));
  int *temp_arr = (int *) malloc(8 * sizeof(int));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  SortBlock64<int, __m256i>(arr, 0);

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}

TEST(SortUtilTest, AVX256SortBlock64Float32BitTest) {
  float *arr;
  aligned_init<float>(arr, 64);
  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);

  auto *check_arr = (float *) malloc(64 * sizeof(float));
  auto *temp_arr = (float *) malloc(8 * sizeof(float));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  SortBlock64<float, __m256>(arr, 0);

  for (int i = 0; i < 64; i++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  free(check_arr);
  free(temp_arr);
}
}