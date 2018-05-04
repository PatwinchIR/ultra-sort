#include "avx512/merge_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"

#ifdef AVX512

namespace avx512 {
TEST(MergeUtilsTest, AVX512MergeRuns16Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init<int>(arr, 256);
  aligned_init<int>(intermediate_arr, 256);

  TestUtil::RandGenInt<int>(arr, 256, -10, 10);

  auto check_arr = (int *) malloc(256 * sizeof(int));
  auto temp_arr = (int *) malloc(16 * sizeof(int));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      intermediate_arr[k * 16 + j] = temp_arr[j];
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 256);

  MergeRuns16<int, __m512i>(intermediate_arr, 256);

  for (int l = 0; l < 256; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergeRuns16Float32BitTest) {
  float *arr, *intermediate_arr;
  aligned_init<float>(arr, 256);
  aligned_init<float>(intermediate_arr, 256);

  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);

  auto *check_arr = (float *) malloc(256 * sizeof(float));
  auto *temp_arr = (float *) malloc(16 * sizeof(float));
  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      intermediate_arr[k * 16 + j] = temp_arr[j];
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  std::sort(check_arr, check_arr + 256);

  MergeRuns16<float, __m512>(intermediate_arr, 256);

  for (int l = 0; l < 256; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergeRuns8Int64BitTest) {
  int64_t *arr, *intermediate_arr;
  aligned_init<int64_t>(arr, 64);
  aligned_init<int64_t>(intermediate_arr, 64);

  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);

  auto check_arr = (int64_t *) malloc(64 * sizeof(int64_t));
  auto temp_arr = (int64_t *) malloc(8 * sizeof(int64_t));
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

  MergeRuns8<int64_t, __m512i>(intermediate_arr, 64);

  for (int l = 0; l < 64; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergeRuns8Float64BitTest) {
  double *arr, *intermediate_arr;
  aligned_init<double>(arr, 64);
  aligned_init<double>(intermediate_arr, 64);

  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);

  auto check_arr = (double *) malloc(64 * sizeof(double));
  auto temp_arr = (double *) malloc(8 * sizeof(double));
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

  MergeRuns8<double, __m512d>(intermediate_arr, 64);

  for (int l = 0; l < 64; ++l) {
    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergePass16Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init<int>(arr, 256);
  aligned_init<int>(intermediate_arr, 256);

  TestUtil::RandGenInt<int>(arr, 256, -10, 10);

  auto check_arr = (int *) malloc(256 * sizeof(int));
  auto temp_arr = (int *) malloc(16 * sizeof(int));

  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      intermediate_arr[k * 16 + j] = temp_arr[j];
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 8; ++l) {
    std::sort(check_arr + l * 32, check_arr + l * 32 + 32);
  }

  int *buffer;
  aligned_init(buffer, 256);
  MergePass16<int, __m512i>(intermediate_arr, buffer, 256, 16);

  for (int m = 0; m < 256; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergePass8Int64BitTest) {
  int64_t *arr, *intermediate_arr;
  aligned_init<int64_t>(arr, 64);
  aligned_init<int64_t>(intermediate_arr, 64);

  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);

  auto check_arr = (int64_t *) malloc(64 * sizeof(int64_t));
  auto temp_arr = (int64_t *) malloc(8 * sizeof(int64_t));

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

  int64_t *buffer;
  aligned_init<int64_t>(buffer, 64);
  MergePass8<int64_t, __m512i>(intermediate_arr, buffer, 64, 8);

  for (int m = 0; m < 64; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergePass8Float64BitTest) {
  double *arr, *intermediate_arr;
  aligned_init<double>(arr, 64);
  aligned_init<double>(intermediate_arr, 64);

  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);

  auto check_arr = (double *) malloc(64 * sizeof(double));
  auto temp_arr = (double *) malloc(8 * sizeof(double));

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

  double *buffer;
  aligned_init<double>(buffer, 64);
  MergePass8<double, __m512d>(intermediate_arr, buffer, 64, 8);

  for (int m = 0; m < 64; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

TEST(MergeUtilsTest, AVX512MergePass16Float32BitTest) {
  float *arr, *intermediate_arr;
  aligned_init<float>(arr, 256);
  aligned_init<float>(intermediate_arr, 256);

  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);

  auto *check_arr = (float *) malloc(256 * sizeof(float));
  auto *temp_arr = (float *) malloc(16 * sizeof(float));

  for (int k = 0; k < 16; k++) {
    for (int i = 0; i < 16; ++i) {
      temp_arr[i] = arr[i * 16 + k];
    }
    std::sort(temp_arr, temp_arr + 16);
    for (int j = 0; j < 16; ++j) {
      intermediate_arr[k * 16 + j] = temp_arr[j];
      check_arr[k * 16 + j] = temp_arr[j];
    }
  }

  for (int l = 0; l < 8; ++l) {
    std::sort(check_arr + l * 32, check_arr + l * 32 + 32);
  }

  float *buffer;
  aligned_init<float>(buffer, 256);
  MergePass16<float, __m512>(intermediate_arr, buffer, 256, 16);

  for (int m = 0; m < 256; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}

}

#endif