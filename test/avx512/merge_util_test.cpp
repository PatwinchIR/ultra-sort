#include "avx512/merge_util.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"

#ifdef AVX512
TEST(MergeUtilsTest, AVX512MergeRuns16Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init<int>(arr, 256);
  aligned_init<int>(intermediate_arr, 256);

  TestUtil::RandGenInt<int>(arr, 256, -10, 10);

  int *check_arr = (int *)malloc(256 * sizeof(int));
  int *temp_arr = (int *)malloc(16 * sizeof(int));
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

  AVX512MergeUtil::MergeRuns16<int,__m512i>(intermediate_arr, 256);

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

  auto *check_arr = (float *)malloc(256 * sizeof(float));
  auto *temp_arr = (float *)malloc(16 * sizeof(float));
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

  AVX512MergeUtil::MergeRuns16<float,__m512>(intermediate_arr, 256);

  for (int l = 0; l < 256; ++l) {
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

  int *check_arr = (int *)malloc(256 * sizeof(int));
  int *temp_arr = (int *)malloc(16 * sizeof(int));

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
  AVX512MergeUtil::MergePass16<int,__m512i>(intermediate_arr, buffer, 256, 16);

  for (int m = 0; m < 256; ++m) {
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

  auto *check_arr = (float *)malloc(256 * sizeof(float));
  auto *temp_arr = (float *)malloc(16 * sizeof(float));

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
  AVX512MergeUtil::MergePass16<float,__m512>(intermediate_arr, buffer, 256, 16);

  for (int m = 0; m < 256; ++m) {
    EXPECT_EQ(check_arr[m], buffer[m]);
  }

  delete[](arr);
  delete[](intermediate_arr);
  delete[](buffer);
  free(check_arr);
  free(temp_arr);
}


#endif