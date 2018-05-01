#include "avx256/merge_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

namespace avx2 {
TEST(MergeUtilsTest, AVX256MergeRuns8Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init<int>(arr, 64);
  aligned_init<int>(intermediate_arr, 64);

  TestUtil::RandGenInt<int>(arr, 64, -10, 10);

  auto *check_arr = (int *) malloc(64 * sizeof(int));
  auto *temp_arr = (int *) malloc(8 * sizeof(int));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      intermediate_arr[k * 8 + j] = temp_arr[j];
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 64);

  MergeRuns8<int, __m256i>(intermediate_arr, 64);

  for (int l = 0; l < 64; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergeRuns8Float32BitTest) {
  float *arr, *intermediate_arr;
  aligned_init<float>(arr, 64);
  aligned_init<float>(intermediate_arr, 64);

  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);

  auto *check_arr = (float *) malloc(64 * sizeof(float));
  auto *temp_arr = (float *) malloc(8 * sizeof(float));
  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      intermediate_arr[k * 8 + j] = temp_arr[j];
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 64);

  MergeRuns8<float, __m256>(intermediate_arr, 64);

  for (int l = 0; l < 64; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergeRuns4Int64BitTest) {
  int64_t *arr, *intermediate_arr;
  aligned_init<int64_t>(arr, 16);
  aligned_init<int64_t>(intermediate_arr, 16);

  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);

  auto *check_arr = (int64_t *) malloc(16 * sizeof(int64_t));
  auto *temp_arr = (int64_t *) malloc(4 * sizeof(int64_t));

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 4; ++i) {
      temp_arr[i] = arr[i * 4 + k];
    }
    std::sort(temp_arr, temp_arr + 4);
    for (int j = 0; j < 4; ++j) {
      intermediate_arr[k * 4 + j] = temp_arr[j];
      check_arr[k * 4 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 16);

  MergeRuns4<int64_t, __m256i>(intermediate_arr, 16);

  for (int l = 0; l < 16; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergeRuns4Float64BitTest) {
  double *arr, *intermediate_arr;
  aligned_init<double>(arr, 16);
  aligned_init<double>(intermediate_arr, 16);

  TestUtil::RandGenFloat<double>(arr, 16, -10, 10);

  auto *check_arr = (double *) malloc(16 * sizeof(double));
  auto *temp_arr = (double *) malloc(4 * sizeof(double));

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 4; ++i) {
      temp_arr[i] = arr[i * 4 + k];
    }
    std::sort(temp_arr, temp_arr + 4);
    for (int j = 0; j < 4; ++j) {
      intermediate_arr[k * 4 + j] = temp_arr[j];
      check_arr[k * 4 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 16);

  MergeRuns4<double, __m256d>(intermediate_arr, 16);

  for (int l = 0; l < 16; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergePass8Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init<int>(arr, 64);
  aligned_init<int>(intermediate_arr, 64);

  TestUtil::RandGenInt<int>(arr, 64, -10, 10);

  int *check_arr = (int *) malloc(64 * sizeof(int));
  int *temp_arr = (int *) malloc(8 * sizeof(int));

  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      intermediate_arr[k * 8 + j] = temp_arr[j];
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 4; ++l) {
    std::sort(check_arr + l * 16, check_arr + l * 16 + 16);
  }

  int *buffer;
  aligned_init<int>(buffer, 64);
  MergePass8<int, __m256i>(intermediate_arr, buffer, 64, 8);

  for (int m = 0; m < 64; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
  free(buffer);
}

TEST(MergeUtilsTest, AVX256MergePass8Float32BitTest) {
  float *arr, *intermediate_arr;
  aligned_init<float>(arr, 64);
  aligned_init<float>(intermediate_arr, 64);

  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);

  auto *check_arr = (float *) malloc(64 * sizeof(float));
  auto *temp_arr = (float *) malloc(8 * sizeof(float));

  for (int k = 0; k < 8; k++) {
    for (int i = 0; i < 8; ++i) {
      temp_arr[i] = arr[i * 8 + k];
    }
    std::sort(temp_arr, temp_arr + 8);
    for (int j = 0; j < 8; ++j) {
      intermediate_arr[k * 8 + j] = temp_arr[j];
      check_arr[k * 8 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 4; ++l) {
    std::sort(check_arr + l * 16, check_arr + l * 16 + 16);
  }

  float *buffer;
  aligned_init<float>(buffer, 64);
  MergePass8<float, __m256>(intermediate_arr, buffer, 64, 8);

  for (int m = 0; m < 64; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergePass4Int64BitTest) {
  int64_t *arr, *intermediate_arr;
  aligned_init<int64_t>(arr, 16);
  aligned_init<int64_t>(intermediate_arr, 16);

  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);

  auto *check_arr = (int64_t *) malloc(16 * sizeof(int64_t));
  auto *temp_arr = (int64_t *) malloc(4 * sizeof(int64_t));

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 4; ++i) {
      temp_arr[i] = arr[i * 4 + k];
    }
    std::sort(temp_arr, temp_arr + 4);
    for (int j = 0; j < 4; ++j) {
      intermediate_arr[k * 4 + j] = temp_arr[j];
      check_arr[k * 4 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 2; ++l) {
    std::sort(check_arr + l * 8, check_arr + l * 8 + 8);
  }

  int64_t *buffer;
  aligned_init<int64_t>(buffer, 16);
  MergePass4<int64_t, __m256i>(intermediate_arr, buffer, 16, 4);

  for (int m = 0; m < 16; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX256MergePass4Float64BitTest) {
  double *arr, *intermediate_arr;
  aligned_init<double>(arr, 16);
  aligned_init<double>(intermediate_arr, 16);

  TestUtil::RandGenFloat<double>(arr, 16, -10, 10);

  auto *check_arr = (double *) malloc(16 * sizeof(double));
  auto *temp_arr = (double *) malloc(4 * sizeof(double));

  for (int k = 0; k < 4; k++) {
    for (int i = 0; i < 4; ++i) {
      temp_arr[i] = arr[i * 4 + k];
    }
    std::sort(temp_arr, temp_arr + 4);
    for (int j = 0; j < 4; ++j) {
      intermediate_arr[k * 4 + j] = temp_arr[j];
      check_arr[k * 4 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 2; ++l) {
    std::sort(check_arr + l * 8, check_arr + l * 8 + 8);
  }

  double *buffer;
  aligned_init<double>(buffer, 16);
  MergePass4<double, __m256d>(intermediate_arr, buffer, 16, 4);

  for (int m = 0; m < 16; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}
}
