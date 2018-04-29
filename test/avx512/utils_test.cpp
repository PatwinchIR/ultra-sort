#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/utils.h"
#include "avx512/sort_util.h"
#include "avx512/merge_util.h"
#include <algorithm>
#include <iterator>

#ifdef AVX512

TEST(UtilsTest, AVX512LoadStore32BitTest) {
  int *a;
  int *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 16);
  TestUtil::PopulateSeqArray(b, 16, 32);
  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::StoreReg(ra, b);
  AVX512Util::StoreReg(rb, a);
  for(int i = 0; i < 16; i++) {
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
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::StoreReg(ra, b);
  AVX512Util::StoreReg(rb, a);
  for(int i = 0; i < 8; i++) {
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
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 16; i++) {
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
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 16; i++) {
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
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
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
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::MinMax8(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512SortBlock256Int32BitTest) {
  int *arr;
  aligned_init(arr, 256);
  TestUtil::RandGenInt(arr, 256, -10, 10);

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

TEST(UtilsTest, AVX512MergeRuns16Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init(arr, 256);
  aligned_init(intermediate_arr, 256);

  TestUtil::RandGenInt(arr, 256, -10, 10);

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

TEST(UtilsTest, AVX512MergePass16Int32BitTest) {
  int *arr, *intermediate_arr;
  aligned_init(arr, 256);
  aligned_init(intermediate_arr, 256);

  TestUtil::RandGenInt(arr, 256, -10, 10);

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
  free(check_arr);
  free(temp_arr);
  free(buffer);
}

TEST(UtilsTest, AVX512BitonicSort16x16Int32BitTest) {
  int *arr;
  aligned_init(arr, 256);
  TestUtil::RandGenInt(arr, 256, -10, 10);
  __m512i r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512Util::BitonicSort16x16(r[0], r[1], r[2], r[3],
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

TEST(UtilsTest, AVX512BitonicSort16x16Float32BitTest) {
  float *arr;
  aligned_init(arr, 256);
  TestUtil::RandGenFloat<float>(arr, 256, -10, 10);
  __m512 r[16];
  for(int i = 0; i < 16; i++) {
    AVX512Util::LoadReg(r[i], arr + i*16);
  }
  AVX512Util::BitonicSort16x16(r[0], r[1], r[2], r[3],
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

TEST(UtilsTest, AVX512BitonicSort8x8Int64BitTest) {
  int64_t *arr;
  aligned_init<int64_t>(arr, 64);
  TestUtil::RandGenInt<int64_t>(arr, 64, -10, 10);
  __m512i r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512Util::BitonicSort8x8(r[0], r[1], r[2], r[3],
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

TEST(UtilsTest, AVX512BitonicSort8x8Float64BitTest) {
  double *arr;
  aligned_init(arr, 64);
  TestUtil::RandGenFloat<double>(arr, 64, -10, 10);
  __m512d r[8];
  for(int i = 0; i < 8; i++) {
    AVX512Util::LoadReg(r[i], arr + i*8);
  }
  AVX512Util::BitonicSort8x8(r[0], r[1], r[2], r[3],
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

TEST(UtilsTest, AVX512BitonicMerge16Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  TestUtil::RandGenInt(a, 16, -10, 10);
  TestUtil::RandGenInt(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  int check_arr[32];
  for (int j = 0; j < 32; ++j) {
    check_arr[j] = j < 16 ? a[j] : b[j - 16];
  }

  std::sort(check_arr, check_arr + 32);

  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);

  for(int i = 0; i < 32; i++) {
    EXPECT_EQ(check_arr[i], i < 16 ? a[i] : b[i - 16]);
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
  AVX512Util::LoadReg(ra, a);
  ra = AVX512Util::Reverse16(ra);
  AVX512Util::StoreReg(ra, a);

  for (int j = 0; j < 16; ++j) {
    EXPECT_EQ(check_arr[j], a[j]);
  }

  delete[](a);
}

TEST(UtilsTest, AVX512IntraRegisterSort16x16Int32BitTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  TestUtil::RandGenInt(a, 16, -10, 10);
  TestUtil::RandGenInt(b, 16, -10, 10);

  std::sort(a, a + 16);
  std::sort(b, b + 16);

  int check_arr[32];
  for (int i = 0; i < 32; ++i) {
    check_arr[i] = i < 16 ? a[i] : b[i - 16];
  }

  std::sort(check_arr, check_arr + 32);

  std::reverse(b, b + 16);

  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::IntraRegisterSort16x16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);

  for (int j = 0; j < 32; ++j) {
    EXPECT_EQ(check_arr[j], j < 16 ? a[j] : b[j - 16]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, FixedTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 16);
  aligned_init<int>(b, 16);

  int temp_a[16] = {-10, -6, -4, -3, -3, -3, -2, 0, 0, 6, 8, 9, 9, 9, 9, 10};
  int temp_b[16] = {7, 6, 4, 4, 3, 3, 3, 2, 1, 0, -5, -6, -8, -8, -9, -9};

  for (int k = 0; k < 16; k ++) {
    a[k] = temp_a[k];
  }

  for (int k = 0; k < 16; k ++) {
    b[k] = temp_b[k];
  }

  int check_arr[32];
  for (int i = 0; i < 32; ++i) {
    check_arr[i] = i < 16 ? temp_a[i] : temp_b[i - 16];
  }

  std::sort(check_arr, check_arr + 32);

  __m512i ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::IntraRegisterSort16x16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);

  for (int j = 0; j < 32; ++j) {
    EXPECT_EQ(check_arr[j], j < 16 ? a[j] : b[j - 16]);
  }

  delete[](a);
  delete[](b);
}

TEST(UtilsTest, AVX512BitonicMerge16Float32BitTest) {
  float *a;
  float *b;
  aligned_init(a, 16);
  aligned_init(b, 16);
  TestUtil::PopulateSeqArray(a, 0, 32, 2);
  TestUtil::PopulateSeqArray(b, 1, 32, 2);
  __m512 ra, rb;
  AVX512Util::LoadReg(ra, a);
  AVX512Util::LoadReg(rb, b);
  AVX512Util::BitonicMerge16(ra, rb);
  AVX512Util::StoreReg(ra, a);
  AVX512Util::StoreReg(rb, b);
  int ab[32];
  for(int i = 0; i < 32; i++) {
    if(i < 16) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i - 16];
    }
    EXPECT_EQ(ab[i], i);
  }
  delete[](a);
  delete[](b);
}

//TEST(UtilsTest, AVX512BitonicMerge8Int64BitTest) {
//  int64_t *a;
//  int64_t *b;
//  aligned_init<int64_t>(a, 8);
//  aligned_init<int64_t>(b, 8);
//  TestUtil::PopulateSeqArray<int64_t>(a, 0, 8, 2);
//  TestUtil::PopulateSeqArray<int64_t>(b, 1, 8, 2);
//  __m512i ra, rb;
//  AVX512Util::LoadReg(ra, a);
//  AVX512Util::LoadReg(rb, b);
//  AVX512Util::BitonicMerge8(ra, rb);
//  AVX512Util::StoreReg(ra, a);
//  AVX512Util::StoreReg(rb, b);
//  int64_t ab[8];
//  for(int i = 0; i < 8; i++) {
//    if(i < 4) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 4];
//    }
//    EXPECT_EQ(ab[i], i);
//  }
//  delete[](a);
//  delete[](b);
//}

//TEST(UtilsTest, AVX512BitonicMerge8Float64BitTest) {
//  double *a;
//  double *b;
//  aligned_init(a, 8);
//  aligned_init(b, 8);
//  TestUtil::PopulateSeqArray(a, 0, 8, 2);
//  TestUtil::PopulateSeqArray(b, 1, 8, 2);
//  __m512d ra, rb;
//  AVX512Util::LoadReg(ra, a);
//  AVX512Util::LoadReg(rb, b);
//  AVX512Util::BitonicMerge8(ra, rb);
//  AVX512Util::StoreReg(ra, a);
//  AVX512Util::StoreReg(rb, b);
//  double ab[8];
//  for(int i = 0; i < 8; i++) {
//    if(i < 4) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 4];
//    }
//    EXPECT_EQ(ab[i], i);
//  }
//  delete[](a);
//  delete[](b);
//}

#endif
