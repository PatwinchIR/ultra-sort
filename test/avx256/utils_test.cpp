#include <avx256/sort_util.h>
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/utils.h"

namespace avx2{
TEST(UtilsTest, AVX256LoadStore32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 8);
  aligned_init(b, 8);
  TestUtil::PopulateSeqArray(a, 0, 8);
  TestUtil::PopulateSeqArray(b, 8, 16);
  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  StoreReg(ra, b);
  StoreReg(rb, a);
  for (int i = 0; i < 8; i++) {
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
  LoadReg(ra, a);
  LoadReg(rb, b);
  StoreReg(ra, b);
  StoreReg(rb, a);
  for (int i = 0; i < 4; i++) {
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
  LoadReg(ra, a);
  LoadReg(rb, b);
  MinMax8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  for (int i = 0; i < 8; i++) {
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
  LoadReg(ra, a);
  LoadReg(rb, b);
  MinMax8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  for (int i = 0; i < 8; i++) {
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
  LoadReg(ra, a);
  LoadReg(rb, b);
  MinMax4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  for (int i = 0; i < 4; i++) {
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
  unsigned int unit_size = 4;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenIntRecords(kv_flat1, unit_size, -10, 10, 0);
  TestUtil::RandGenIntRecords(kv_flat2, unit_size, -10, 10, unit_size * 2);
  std::map<T, T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T, T>(kv_flat1[2 * i + 1], kv_flat1[2 * i]));
    kv_map.insert(std::pair<T, T>(kv_flat2[2 * i + 1], kv_flat2[2 * i]));
  }

  __m256i r1, r2;
  LoadReg(r1, kv_flat1);
  LoadReg(r2, kv_flat2);
  MaskedMinMax8(r1, r2);
  StoreReg(r1, kv_flat1);
  StoreReg(r2, kv_flat2);
  for (int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2 * i], kv_flat2[2 * i]);
    EXPECT_EQ(kv_flat1[2 * i], kv_map[kv_flat1[2 * i + 1]]);
    EXPECT_EQ(kv_flat2[2 * i], kv_map[kv_flat2[2 * i + 1]]);
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
  unsigned int unit_size = 4;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenFloatRecords(kv_flat1, unit_size, -10.0f, 10.0f, 0);
  TestUtil::RandGenFloatRecords(kv_flat2, unit_size, -10.0f, 10.0f, unit_size * 2);
  std::map<T, T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T, T>(kv_flat1[2 * i + 1], kv_flat1[2 * i]));
    kv_map.insert(std::pair<T, T>(kv_flat2[2 * i + 1], kv_flat2[2 * i]));
  }

  __m256 r1, r2;
  LoadReg(r1, kv_flat1);
  LoadReg(r2, kv_flat2);
  MaskedMinMax8(r1, r2);
  StoreReg(r1, kv_flat1);
  StoreReg(r2, kv_flat2);
  for (int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2 * i], kv_flat2[2 * i]);
    EXPECT_EQ(kv_flat1[2 * i], kv_map[kv_flat1[2 * i + 1]]);
    EXPECT_EQ(kv_flat2[2 * i], kv_map[kv_flat2[2 * i + 1]]);
  }
  delete[](kv_flat1);
  delete[](kv_flat2);
}

TEST(UtilsTest, AVX256MaskedMinMax4Int64Test) {
  // For masked tests, we assume k,v follow one another with native type
  // i.e. for 32-bit int k,v array -> Assume its flattened out to an int array
  // 4 32-bit key-value integers will fit in AVX2 register
  // This test is currently weak - only checks for sorted order
  using T = int64_t;
  unsigned int unit_size = 2;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenIntRecords(kv_flat1, unit_size*2, -10ll, 10ll, 0u);
  TestUtil::RandGenIntRecords(kv_flat2, unit_size*2, -10ll, 10ll, unit_size*2u);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T,T>(kv_flat1[2*i + 1], kv_flat1[2*i]));
    kv_map.insert(std::pair<T,T>(kv_flat2[2*i + 1], kv_flat2[2*i]));
  }

  __m256i r1, r2;
  LoadReg(r1, kv_flat1);
  LoadReg(r2, kv_flat2);
  MaskedMinMax4(r1, r2);
  StoreReg(r1, kv_flat1);
  StoreReg(r2, kv_flat2);
  for(int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2*i], kv_flat2[2*i]);
    EXPECT_EQ(kv_flat1[2*i], kv_map[kv_flat1[2*i + 1]]);
    EXPECT_EQ(kv_flat2[2*i], kv_map[kv_flat2[2*i + 1]]);
  }
  delete[](kv_flat1);
  delete[](kv_flat2);
}

TEST(UtilsTest, AVX256MaskedMinMax4Float64Test) {
  // For masked tests, we assume k,v follow one another with native type
  // i.e. for 32-bit int k,v array -> Assume its flattened out to an int array
  // 4 32-bit key-value integers will fit in AVX2 register
  // This test is currently weak - only checks for sorted order
  using T = double;
  unsigned int unit_size = 2;
  T *kv_flat1, *kv_flat2;
  TestUtil::RandGenFloatRecords(kv_flat1, unit_size*2, -10.0, 10.0, 0);
  TestUtil::RandGenFloatRecords(kv_flat2, unit_size*2, -10.0, 10.0, unit_size*2);
  std::map<T,T> kv_map;
  for (int i = 0; i < unit_size; ++i) {
    kv_map.insert(std::pair<T,T>(kv_flat1[2*i + 1], kv_flat1[2*i]));
    kv_map.insert(std::pair<T,T>(kv_flat2[2*i + 1], kv_flat2[2*i]));
  }

  __m256d r1, r2;
  LoadReg(r1, kv_flat1);
  LoadReg(r2, kv_flat2);
  MaskedMinMax4(r1, r2);
  StoreReg(r1, kv_flat1);
  StoreReg(r2, kv_flat2);
  for(int i = 0; i < unit_size; i++) {
    EXPECT_LE(kv_flat1[2*i], kv_flat2[2*i]);
    EXPECT_EQ(kv_flat1[2*i], kv_map[kv_flat1[2*i + 1]]);
    EXPECT_EQ(kv_flat2[2*i], kv_map[kv_flat2[2*i + 1]]);
  }
  delete[](kv_flat1);
  delete[](kv_flat2);
}

TEST(UtilsTest, AVX256BitonicSort8x8Int32BitTest) {
  int *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenInt(arr, 64, -10, 10);
  __m256i r[8];
  for (int i = 0; i < 8; i++) {
    LoadReg(r[i], arr + i * 8);
  }
  BitonicSort8x8(r[0], r[1], r[2], r[3],
                 r[4], r[5], r[6], r[7]);
  for (int i = 0; i < 8; i++) {
    StoreReg(r[i], arr + i * 8);
  }
  for (int i = 8; i < 64; i += 8) {
    for (int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j - 8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256BitonicSort8x8Float32BitTest) {
  float *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);
  __m256 r[8];
  for (int i = 0; i < 8; i++) {
    LoadReg(r[i], arr + i * 8);
  }
  BitonicSort8x8(r[0], r[1], r[2], r[3],
                 r[4], r[5], r[6], r[7]);
  for (int i = 0; i < 8; i++) {
    StoreReg(r[i], arr + i * 8);
  }
  for (int i = 8; i < 64; i += 8) {
    for (int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j - 8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256MaskedBitonicSort4x8Int32BitTest) {
  using T = int;
  T *arr;
  unsigned int unit_size = 4;
  int num_cols = unit_size * 2;
  int num_rows = unit_size;
  TestUtil::RandGenIntRecords(arr, num_cols * num_rows, -10, 10);
  std::map<T, T> kv_map;
  for (int i = 0; i < unit_size * num_rows; ++i) {
    kv_map.insert(std::pair<T, T>(arr[2 * i + 1], arr[2 * i]));
  }

  __m256i r[num_rows];
  for (int i = 0; i < num_rows; i++) {
    LoadReg(r[i], arr + i * num_cols);
  }
  MaskedBitonicSort4x8(r[0], r[1], r[2], r[3]);
  for (int i = 0; i < num_rows; i++) {
    StoreReg(r[i], arr + i * num_cols);
  }

  for (int i = num_cols; i < num_rows * num_cols; i += num_cols) {
    for (int j = i; j < i + num_cols; j += 2) {
      EXPECT_LE(arr[j - 8], arr[j]);
    }
  }

  for (int i = 0; i < unit_size * num_rows; ++i) {
    EXPECT_EQ(kv_map[arr[2 * i + 1]], arr[2 * i]);
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256MaskedBitonicSort8x8Float32BitTest) {
  using T = float;
  T *arr;
  unsigned int unit_size = 4;
  int num_cols = unit_size * 2;
  int num_rows = unit_size;
  TestUtil::RandGenFloatRecords(arr, num_cols * num_rows, -10.0f, 10.0f);
  std::map<T, T> kv_map;
  for (int i = 0; i < unit_size * num_rows; ++i) {
    kv_map.insert(std::pair<T, T>(arr[2 * i + 1], arr[2 * i]));
  }

  __m256 r[num_rows];
  for (int i = 0; i < num_rows; i++) {
    LoadReg(r[i], arr + i * num_cols);
  }
  MaskedBitonicSort4x8(r[0], r[1], r[2], r[3]);
  for (int i = 0; i < num_rows; i++) {
    StoreReg(r[i], arr + i * num_cols);
  }

  for (int i = num_cols; i < num_rows * num_cols; i += num_cols) {
    for (int j = i; j < i + num_cols; j += 2) {
      EXPECT_LE(arr[j - 8], arr[j]);
    }
  }

  for (int i = 0; i < unit_size * num_rows; ++i) {
    EXPECT_EQ(kv_map[arr[2 * i + 1]], arr[2 * i]);
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256BitonicSort4x4Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 16);
  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);
  __m256i r[4];
  for (int i = 0; i < 4; i++) {
    LoadReg(r[i], arr + i * 4);
  }
  BitonicSort4x4(r[0], r[1], r[2], r[3]);
  for (int i = 0; i < 4; i++) {
    StoreReg(r[i], arr + i * 4);
  }
  for (int i = 4; i < 16; i += 4) {
    for (int j = i; j < i + 4; j++) {
      EXPECT_LE(arr[j - 4], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256BitonicSort4x4Float64BitTest) {
  double *arr;
  aligned_init<double>(arr, 16);
  TestUtil::RandGenFloat<double>(arr, 16, -10, 10);
  __m256d r[4];
  for (int i = 0; i < 4; i++) {
    LoadReg(r[i], arr + i * 4);
  }
  BitonicSort4x4(r[0], r[1], r[2], r[3]);
  for (int i = 0; i < 4; i++) {
    StoreReg(r[i], arr + i * 4);
  }
  for (int i = 4; i < 16; i += 4) {
    for (int j = i; j < i + 4; j++) {
      EXPECT_LE(arr[j - 4], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX256IntraRegisterSort8x8Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);

  TestUtil::RandGenInt<int>(a, 8, -10, 10);
  TestUtil::RandGenInt<int>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  int check_arr[16];
  for (int i = 0; i < 16; ++i) {
    check_arr[i] = i < 8 ? a[i] : b[i - 8];
  }

  std::sort(check_arr, check_arr + 16);

  std::reverse(b, b + 8);

  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort8x8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(check_arr[j], j < 8 ? a[j] : b[j - 8]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256IntraRegisterSort8x8Float32BitTest) {
  float *a;
  float *b;
  aligned_init<float>(a, 8);
  aligned_init<float>(b, 8);

  TestUtil::RandGenFloat<float>(a, 8, -10, 10);
  TestUtil::RandGenFloat<float>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  float check_arr[16];
  for (int i = 0; i < 16; ++i) {
    check_arr[i] = i < 8 ? a[i] : b[i - 8];
  }

  std::sort(check_arr, check_arr + 16);

  std::reverse(b, b + 8);

  __m256 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort8x8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(check_arr[j], j < 8 ? a[j] : b[j - 8]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256IntraRegisterSort4x4Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 4);
  aligned_init<int64_t>(b, 4);

  TestUtil::RandGenInt<int64_t>(a, 4, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 4, -10, 10);

  std::sort(a, a + 4);
  std::sort(b, b + 4);

  int64_t check_arr[8];
  for (int i = 0; i < 8; ++i) {
    check_arr[i] = i < 4 ? a[i] : b[i - 4];
  }

  std::sort(check_arr, check_arr + 8);

  std::reverse(b, b + 4);

  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort4x4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 8; ++j) {
    EXPECT_EQ(check_arr[j], j < 4 ? a[j] : b[j - 4]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256IntraRegisterSort4x4Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 4);
  aligned_init<double>(b, 4);

  TestUtil::RandGenFloat<double>(a, 4, -10, 10);
  TestUtil::RandGenFloat<double>(b, 4, -10, 10);

  std::sort(a, a + 4);
  std::sort(b, b + 4);

  double check_arr[8];
  for (int i = 0; i < 8; ++i) {
    check_arr[i] = i < 4 ? a[i] : b[i - 4];
  }

  std::sort(check_arr, check_arr + 8);

  std::reverse(b, b + 4);

  __m256d ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort4x4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 8; ++j) {
    EXPECT_EQ(check_arr[j], j < 4 ? a[j] : b[j - 4]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256BitonicMerge8Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);

  TestUtil::RandGenInt<int>(a, 8, -10, 10);
  TestUtil::RandGenInt<int>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  int check_arr[16];
  for (int j = 0; j < 16; ++j) {
    check_arr[j] = j < 8 ? a[j] : b[j - 8];
  }

  std::sort(check_arr, check_arr + 16);

  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(check_arr[i], i < 8 ? a[i] : b[i - 8]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256BitonicMerge8Float32BitTest) {
  float *a;
  float *b;
  aligned_init<float>(a, 8);
  aligned_init<float>(b, 8);

  TestUtil::RandGenFloat<float>(a, 8, -10, 10);
  TestUtil::RandGenFloat<float>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  float check_arr[16];
  for (int j = 0; j < 16; ++j) {
    check_arr[j] = j < 8 ? a[j] : b[j - 8];
  }

  std::sort(check_arr, check_arr + 16);

  __m256 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge8(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(check_arr[i], i < 8 ? a[i] : b[i - 8]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256BitonicMerge4Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 4);
  aligned_init<int64_t>(b, 4);

  TestUtil::RandGenInt<int64_t>(a, 4, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 4, -10, 10);

  std::sort(a, a + 4);
  std::sort(b, b + 4);

  int64_t check_arr[8];
  for (int j = 0; j < 8; ++j) {
    check_arr[j] = j < 4 ? a[j] : b[j - 4];
  }

  std::sort(check_arr, check_arr + 8);

  __m256i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(check_arr[i], i < 4 ? a[i] : b[i - 4]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX256BitonicMerge4Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 4);
  aligned_init<double>(b, 4);

  TestUtil::RandGenFloat<double>(a, 4, -10, 10);
  TestUtil::RandGenFloat<double>(b, 4, -10, 10);

  std::sort(a, a + 4);
  std::sort(b, b + 4);

  double check_arr[8];
  for (int j = 0; j < 8; ++j) {
    check_arr[j] = j < 4 ? a[j] : b[j - 4];
  }

  std::sort(check_arr, check_arr + 8);

  __m256d ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge4(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 8; i++) {
    EXPECT_EQ(check_arr[i], i < 4 ? a[i] : b[i - 4]);
  }
  delete[](a);
  delete[](b);
}

}