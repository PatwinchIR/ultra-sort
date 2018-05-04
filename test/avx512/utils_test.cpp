#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"
#include "avx512/sort_util.h"
#include "avx512/merge_util.h"
#include <algorithm>
#include <iterator>

#ifdef AVX512
namespace avx512 {
TEST(UtilsTest, AVX512LoadStore32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 16);
  TestUtil::PopulateSeqArray(b, 16, 32);
  __m512i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  StoreReg(ra, b);
  StoreReg(rb, a);
  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 16);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512LoadStore64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);
  TestUtil::PopulateSeqArray<int64_t>(a, 0, 8);
  TestUtil::PopulateSeqArray<int64_t>(b, 8, 16);
  __m512i ra, rb;
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

TEST(UtilsTest, AVX512MinMax16Int32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::RandGenInt(a, 16, -10, 10);
  TestUtil::RandGenInt(b, 16, -10, 10);
  __m512i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  MinMax16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  for (int i = 0; i < 16; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax16Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  float lo = -10;
  float hi = 10;
  TestUtil::RandGenFloat(a, 16, lo, hi);
  TestUtil::RandGenFloat(b, 16, lo, hi);
  __m512 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  MinMax16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);
  for (int i = 0; i < 16; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512MinMax8Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);
  TestUtil::RandGenInt<int64_t>(a, 8, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 8, -10, 10);
  __m512i ra, rb;
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

TEST(UtilsTest, AVX512MinMax8Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 8);
  aligned_init<double>(b, 8);
  TestUtil::RandGenFloat<double>(a, 8, -10, 10);
  TestUtil::RandGenFloat<double>(b, 8, -10, 10);
  __m512d ra, rb;
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

TEST(UtilsTest, AVX512MaskedMinMax16Int32BitTest_Fixed) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  int temp_a[16] = {3, 0, 2, 2, 5, 9, 1, -1, 8, 7, 4, 3, 7, 4, 0, 9};
  int temp_b[16] = {1, 1, 2, 2, 7, 7, 0, 0, 9, 9, 5, 5, 3, 3, 4, 4};

//  int temp_a[16] = {3, 3, 2, 2, 5, 5, 1, 1, 8, 8, 4, 4, 7, 7, 0, 0};
//  int temp_b[16] = {1, 1, 2, 2, 7, 7, 0, 0, 9, 9, 5, 5, 3, 3, 4, 4};

  for (int i = 0; i < 16; i ++) {
    a[i] = temp_a[i];
  }

  for (int i = 0; i < 16; i ++) {
    b[i] = temp_b[i];
  }

  print_arr(a, 0, 16, "a: ");
  print_arr(b, 0, 16, "b: ");

  __m512i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  MaskedMinMax16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  int check_arr_min[16] = {1, 1, 2, 2, 5, 9, 0, 0, 8, 7, 4, 3, 3, 3, 0, 9};
  int check_arr_max[16] = {3, 0, 2, 2, 7, 7, 1, -1, 9, 9, 5, 5, 7, 4, 4, 4};

  print_arr(a, 0, 16, "min: ");
  print_arr(check_arr_min, 0, 16, "check min: ");
  print_arr(b, 0, 16, "max: ");
  print_arr(check_arr_max, 0, 16, "check max: ");

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(check_arr_min[i], a[i]);
  }

  for (int i = 0; i < 16; i++) {
    EXPECT_EQ(check_arr_max[i], b[i]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512Reverse16Int32BitTest) {
  int *a;
  aligned_init<int>(a, 16);

  TestUtil::RandGenInt(a, 16, -10, 10);
  std::sort(a, a + 16);

  int check_arr[16];
  for (int i = 15; i >= 0; --i) {
    check_arr[15 - i] = a[i];
  }

  __m512i ra;
  LoadReg(ra, a);
  Reverse16(ra);
  StoreReg(ra, a);

  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(check_arr[j], a[j]);
  }

  delete[](a);
}

TEST(UtilsTest, AVX512BitonicSort16x16Int32BitTest) {
  int *arr;
  aligned_init<int>(arr, 256);
  TestUtil::RandGenInt<int>(arr, 256, -10, 10);
  __m512i r[16];
  for (int i = 0; i < 16; i++) {
    LoadReg(r[i], arr + i * 16);
  }
  BitonicSort16x16(r[0], r[1], r[2], r[3],
                               r[4], r[5], r[6], r[7],
                               r[8], r[9], r[10], r[11],
                               r[12], r[13], r[14], r[15]);
  for (int i = 0; i < 16; i++) {
    StoreReg(r[i], arr + i * 16);
  }
  for (int i = 16; i < 256; i += 16) {
    for (int j = i; j < i + 16; j++) {
      EXPECT_LE(arr[j - 16], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicSort16x16Float32BitTest) {
  float *arr;
  aligned_init<float>(arr, 256);
  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);
  __m512 r[16];
  for (int i = 0; i < 16; i++) {
    LoadReg(r[i], arr + i * 16);
  }
  BitonicSort16x16(r[0], r[1], r[2], r[3],
                               r[4], r[5], r[6], r[7],
                               r[8], r[9], r[10], r[11],
                               r[12], r[13], r[14], r[15]);
  for (int i = 0; i < 16; i++) {
    StoreReg(r[i], arr + i * 16);
  }
  for (int i = 16; i < 256; i += 16) {
    for (int j = i; j < i + 16; j++) {
      EXPECT_LE(arr[j - 16], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, AVX512BitonicSort8x8Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 64);
  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);
  __m512i r[8];
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

TEST(UtilsTest, AVX512Transpose8x8Int64BitTest) {
  int64_t *arr, *check_arr;
  aligned_init<int64_t>(arr, 64);
  aligned_init<int64_t>(check_arr, 64);
  for (int i = 0; i < 64; i ++) {
    arr[i] = i;
    check_arr[i] = i / 8 + (i % 8) * 8;
  }

  __m512i r[8];
  for (int i = 0; i < 8; i++) {
    LoadReg(r[i], arr + i * 8);
  }
  Transpose8x8(r[0], r[1], r[2], r[3],
               r[4], r[5], r[6], r[7]);
  for (int i = 0; i < 8; i++) {
    StoreReg(r[i], arr + i * 8);
  }

  for (int i = 0; i < 64; i ++) {
    EXPECT_EQ(check_arr[i], arr[i]);
  }

  delete[](arr);
  delete[](check_arr);
}

TEST(UtilsTest, AVX512BitonicSort8x8Float64BitTest) {
  double *arr;
  aligned_init<double>(arr, 64);
  TestUtil::RandGenFloat<double>(arr, 64, -10.0, 10.0);
  __m512d r[8];
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

TEST(UtilsTest, AVX512IntraRegisterSort16x16Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  TestUtil::RandGenInt<int>(a, 16, -10, 10);
  TestUtil::RandGenInt<int>(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  int check_arr[32];
  for (int i = 0; i < 32; ++i) {
    check_arr[i] = i < 16 ? a[i] : b[i - 16];
  }

  std::sort(check_arr, check_arr + 32);

  std::reverse(b, b + 16);

  __m512i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort16x16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 32; ++j) {
    EXPECT_EQ(check_arr[j], j < 16 ? a[j] : b[j - 16]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512IntraRegisterSort16x16Float32BitTest) {
  float *a;
  float *b;
  aligned_init<float>(a, 16);
  aligned_init<float>(b, 16);

  TestUtil::RandGenFloat<float>(a, 16, -10, 10);
  TestUtil::RandGenFloat<float>(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  float check_arr[32];
  for (int i = 0; i < 32; ++i) {
    check_arr[i] = i < 16 ? a[i] : b[i - 16];
  }

  std::sort(check_arr, check_arr + 32);

  std::reverse(b, b + 16);

  __m512 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  IntraRegisterSort16x16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int j = 0; j < 32; ++j) {
    EXPECT_EQ(check_arr[j], j < 16 ? a[j] : b[j - 16]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512IntraRegisterSort8x8Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);

  TestUtil::RandGenInt<int64_t>(a, 8, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  int64_t check_arr[16];
  for (int i = 0; i < 16; ++i) {
    check_arr[i] = i < 8 ? a[i] : b[i - 8];
  }

  std::sort(check_arr, check_arr + 16);

  std::reverse(b, b + 8);

  __m512i ra, rb;
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

TEST(UtilsTest, AVX512IntraRegisterSort8x8Int64BitTest_Fixed) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);

  int64_t temp_a[8] = {1, 3, 5, 7, 9, 11, 13, 15};
  int64_t temp_b[8] = {14, 12, 10, 8, 6, 4, 2, 0};
  
  for (int i = 0; i < 8; i ++) {
    a[i] = temp_a[i];
  }
  
  for (int i = 0; i < 8; i ++) {
    b[i] = temp_b[i];
  }

  int64_t check_arr[16];
  for (int i = 0; i < 16; ++i) {
    check_arr[i] = i;
  }

  __m512i ra, rb;
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

TEST(UtilsTest, AVX512IntraRegisterSort8x8Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 8);
  aligned_init<double>(b, 8);

  TestUtil::RandGenFloat<double>(a, 8, -10, 10);
  TestUtil::RandGenFloat<double>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  double check_arr[16];
  for (int i = 0; i < 16; ++i) {
    check_arr[i] = i < 8 ? a[i] : b[i - 8];
  }

  std::sort(check_arr, check_arr + 16);

  std::reverse(b, b + 8);

  __m512d ra, rb;
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

TEST(UtilsTest, AVX512BitonicMerge16Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  TestUtil::RandGenInt<int>(a, 16, -10, 10);
  TestUtil::RandGenInt<int>(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  int check_arr[32];
  for (int j = 0; j < 32; ++j) {
    check_arr[j] = j < 16 ? a[j] : b[j - 16];
  }

  std::sort(check_arr, check_arr + 32);

  __m512i ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(check_arr[i], i < 16 ? a[i] : b[i - 16]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge16Float32BitTest) {
  float *a;
  float *b;
  aligned_init<float>(a, 16);
  aligned_init<float>(b, 16);

  TestUtil::RandGenFloat<float>(a, 16, -10, 10);
  TestUtil::RandGenFloat<float>(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  float check_arr[32];
  for (int j = 0; j < 32; ++j) {
    check_arr[j] = j < 16 ? a[j] : b[j - 16];
  }

  std::sort(check_arr, check_arr + 32);

  __m512 ra, rb;
  LoadReg(ra, a);
  LoadReg(rb, b);
  BitonicMerge16(ra, rb);
  StoreReg(ra, a);
  StoreReg(rb, b);

  for (int i = 0; i < 32; i++) {
    EXPECT_EQ(check_arr[i], i < 16 ? a[i] : b[i - 16]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge8Int64BitTest) {
  int64_t *a;
  int64_t *b;
  aligned_init<int64_t>(a, 8);
  aligned_init<int64_t>(b, 8);

  TestUtil::RandGenInt<int64_t>(a, 8, -10, 10);
  TestUtil::RandGenInt<int64_t>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  int64_t check_arr[16];
  for (int j = 0; j < 16; ++j) {
    check_arr[j] = j < 8 ? a[j] : b[j - 8];
  }

  std::sort(check_arr, check_arr + 16);

  __m512i ra, rb;
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

TEST(UtilsTest, AVX512BitonicMerge8Float64BitTest) {
  double *a;
  double *b;
  aligned_init<double>(a, 8);
  aligned_init<double>(b, 8);

  TestUtil::RandGenFloat<double>(a, 8, -10, 10);
  TestUtil::RandGenFloat<double>(b, 8, -10, 10);

  std::sort(a, a + 8);
  std::sort(b, b + 8);

  double check_arr[16];
  for (int j = 0; j < 16; ++j) {
    check_arr[j] = j < 8 ? a[j] : b[j - 8];
  }

  std::sort(check_arr, check_arr + 16);

  __m512d ra, rb;
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

}

#endif
