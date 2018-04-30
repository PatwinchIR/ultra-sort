//#include "avx256/merge_util.h"
//#include "gtest/gtest.h"
//#include "test_util.h"
//#include "avx256/utils.h"
//
//namespace avx2 {
//TEST(MergeUtilsTest, AVX256BitonicMerge8Int32BitTest) {
//  int *a;
//  int *b;
//  aligned_init<int>(a, 8);
//  aligned_init<int>(b, 8);
//
//  TestUtil::RandGenInt<int>(a, 8, -10, 10);
//  TestUtil::RandGenInt<int>(b, 8, -10, 10);
//
//  std::sort(a, a + 8);
//  std::sort(b, b + 8);
//
//  int check_arr[16];
//  for (int j = 0; j < 16; ++j) {
//    check_arr[j] = j < 8 ? a[j] : b[j - 8];
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  __m256i ra, rb;
//<<<<<<< HEAD
//  LoadReg(ra, a);
//  LoadReg(rb, b);
//  BitonicMerge8(ra, rb);
//  StoreReg(ra, a);
//  StoreReg(rb, b);
//  int ab[16];
//  for (int i = 0; i < 16; i++) {
//    if (i < 8) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 8];
//    }
//    EXPECT_EQ(ab[i], i);
//=======
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::BitonicMerge8(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for(int i = 0; i < 16; i++) {
//    EXPECT_EQ(check_arr[i], i < 8 ? a[i] : b[i - 8]);
//>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
//  }
//  delete[](a);
//  delete[](b);
//}
//
//TEST(MergeUtilsTest, AVX256BitonicMerge8Float32BitTest) {
//  float *a;
//  float *b;
//  aligned_init<float>(a, 8);
//  aligned_init<float>(b, 8);
//
//  TestUtil::RandGenFloat<float>(a, 8, -10, 10);
//  TestUtil::RandGenFloat<float>(b, 8, -10, 10);
//
//  std::sort(a, a + 8);
//  std::sort(b, b + 8);
//
//  float check_arr[16];
//  for (int j = 0; j < 16; ++j) {
//    check_arr[j] = j < 8 ? a[j] : b[j - 8];
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  __m256 ra, rb;
//<<<<<<< HEAD
//  LoadReg(ra, a);
//  LoadReg(rb, b);
//  BitonicMerge8(ra, rb);
//  StoreReg(ra, a);
//  StoreReg(rb, b);
//  float ab[16];
//  for (int i = 0; i < 16; i++) {
//    if (i < 8) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 8];
//    }
//    EXPECT_EQ(ab[i], i);
//  }
//  delete (a);
//  delete (b);
//=======
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::BitonicMerge8(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for(int i = 0; i < 16; i++) {
//    EXPECT_EQ(check_arr[i], i < 8 ? a[i] : b[i - 8]);
//  }
//  delete[](a);
//  delete[](b);
//>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
//}
//
//TEST(MergeUtilsTest, AVX256BitonicMerge4Int64BitTest) {
//  int64_t *a;
//  int64_t *b;
//  aligned_init<int64_t>(a, 4);
//  aligned_init<int64_t>(b, 4);
//
//  TestUtil::RandGenInt<int64_t>(a, 4, -10, 10);
//  TestUtil::RandGenInt<int64_t>(b, 4, -10, 10);
//
//  std::sort(a, a + 4);
//  std::sort(b, b + 4);
//
//  int64_t check_arr[8];
//  for (int j = 0; j < 8; ++j) {
//    check_arr[j] = j < 4 ? a[j] : b[j - 4];
//  }
//
//  std::sort(check_arr, check_arr + 8);
//
//  __m256i ra, rb;
//<<<<<<< HEAD
//  LoadReg(ra, a);
//  LoadReg(rb, b);
//  BitonicMerge4(ra, rb);
//  StoreReg(ra, a);
//  StoreReg(rb, b);
//  int64_t ab[8];
//  for (int i = 0; i < 8; i++) {
//    if (i < 4) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 4];
//    }
//    EXPECT_EQ(ab[i], i);
//=======
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::BitonicMerge4(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for(int i = 0; i < 8; i++) {
//    EXPECT_EQ(check_arr[i], i < 4 ? a[i] : b[i - 4]);
//>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
//  }
//  delete[](a);
//  delete[](b);
//}
//
//TEST(MergeUtilsTest, AVX256BitonicMerge4Float64BitTest) {
//  double *a;
//  double *b;
//  aligned_init<double>(a, 4);
//  aligned_init<double>(b, 4);
//
//  TestUtil::RandGenFloat<double>(a, 4, -10, 10);
//  TestUtil::RandGenFloat<double>(b, 4, -10, 10);
//
//  std::sort(a, a + 4);
//  std::sort(b, b + 4);
//
//  double check_arr[8];
//  for (int j = 0; j < 8; ++j) {
//    check_arr[j] = j < 4 ? a[j] : b[j - 4];
//  }
//
//  std::sort(check_arr, check_arr + 8);
//
//  __m256d ra, rb;
//<<<<<<< HEAD
//  LoadReg(ra, a);
//  LoadReg(rb, b);
//  BitonicMerge4(ra, rb);
//  StoreReg(ra, a);
//  StoreReg(rb, b);
//  double ab[8];
//  for (int i = 0; i < 8; i++) {
//    if (i < 4) {
//      ab[i] = a[i];
//    } else {
//      ab[i] = b[i - 4];
//=======
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::BitonicMerge4(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for(int i = 0; i < 8; i++) {
//    EXPECT_EQ(check_arr[i], i < 4 ? a[i] : b[i - 4]);
//  }
//  delete[](a);
//  delete[](b);
//}
//
//TEST(MergeUtilsTest, AVX256MergeRuns8Int32BitTest) {
//  int *arr, *intermediate_arr;
//  aligned_init<int>(arr, 64);
//  aligned_init<int>(intermediate_arr, 64);
//
//  TestUtil::RandGenInt<int>(arr, 64, -10, 10);
//
//  auto *check_arr = (int *)malloc(64 * sizeof(int));
//  auto *temp_arr = (int *)malloc(8 * sizeof(int));
//  for (int k = 0; k < 8; k++) {
//    for (int i = 0; i < 8; ++i) {
//      temp_arr[i] = arr[i * 8 + k];
//    }
//    std::sort(temp_arr, temp_arr + 8);
//    for (int j = 0; j < 8; ++j) {
//      intermediate_arr[k * 8 + j] = temp_arr[j];
//      check_arr[k * 8 + j] = temp_arr[j];
//    }
//  }
//
//  std::sort(check_arr, check_arr + 64);
//
//  AVX256MergeUtil::MergeRuns8<int,__m512i>(intermediate_arr, 64);
//
//  for (int l = 0; l < 64; ++l) {
//    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergeRuns8Float32BitTest) {
//  float *arr, *intermediate_arr;
//  aligned_init<float>(arr, 64);
//  aligned_init<float>(intermediate_arr, 64);
//
//  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);
//
//  auto *check_arr = (float *)malloc(64 * sizeof(float));
//  auto *temp_arr = (float *)malloc(8 * sizeof(float));
//  for (int k = 0; k < 8; k++) {
//    for (int i = 0; i < 8; ++i) {
//      temp_arr[i] = arr[i * 8 + k];
//    }
//    std::sort(temp_arr, temp_arr + 8);
//    for (int j = 0; j < 8; ++j) {
//      intermediate_arr[k * 8 + j] = temp_arr[j];
//      check_arr[k * 8 + j] = temp_arr[j];
//    }
//  }
//
//  std::sort(check_arr, check_arr + 64);
//
//  AVX256MergeUtil::MergeRuns8<float,__m512>(intermediate_arr, 64);
//
//  for (int l = 0; l < 64; ++l) {
//    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergeRuns4Int64BitTest) {
//  int64_t *arr, *intermediate_arr;
//  aligned_init<int64_t>(arr, 16);
//  aligned_init<int64_t>(intermediate_arr, 16);
//
//  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);
//
//  auto *check_arr = (int64_t *)malloc(16 * sizeof(int64_t));
//  auto *temp_arr = (int64_t *)malloc(4 * sizeof(int64_t));
//
//  for (int k = 0; k < 4; k++) {
//    for (int i = 0; i < 4; ++i) {
//      temp_arr[i] = arr[i * 4 + k];
//    }
//    std::sort(temp_arr, temp_arr + 4);
//    for (int j = 0; j < 4; ++j) {
//      intermediate_arr[k * 4 + j] = temp_arr[j];
//      check_arr[k * 4 + j] = temp_arr[j];
//    }
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  AVX256MergeUtil::MergeRuns4<int64_t,__m512i>(intermediate_arr, 16);
//
//  for (int l = 0; l < 16; ++l) {
//    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergeRuns4Float64BitTest) {
//  double *arr, *intermediate_arr;
//  aligned_init<double>(arr, 16);
//  aligned_init<double>(intermediate_arr, 16);
//
//  TestUtil::RandGenFloat<double>(arr, 16, -10, 10);
//
//  auto *check_arr = (double *)malloc(16 * sizeof(double));
//  auto *temp_arr = (double *)malloc(4 * sizeof(double));
//
//  for (int k = 0; k < 4; k++) {
//    for (int i = 0; i < 4; ++i) {
//      temp_arr[i] = arr[i * 4 + k];
//    }
//    std::sort(temp_arr, temp_arr + 4);
//    for (int j = 0; j < 4; ++j) {
//      intermediate_arr[k * 4 + j] = temp_arr[j];
//      check_arr[k * 4 + j] = temp_arr[j];
//    }
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  AVX256MergeUtil::MergeRuns4<double,__m512>(intermediate_arr, 16);
//
//  for (int l = 0; l < 16; ++l) {
//    EXPECT_EQ(check_arr[l], intermediate_arr[l]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergePass8Int32BitTest) {
//  int *arr, *intermediate_arr;
//  aligned_init<int>(arr, 64);
//  aligned_init<int>(intermediate_arr, 64);
//
//  TestUtil::RandGenInt<int>(arr, 64, -10, 10);
//
//  int *check_arr = (int *)malloc(64 * sizeof(int));
//  int *temp_arr = (int *)malloc(8 * sizeof(int));
//
//  for (int k = 0; k < 8; k++) {
//    for (int i = 0; i < 8; ++i) {
//      temp_arr[i] = arr[i * 8 + k];
//    }
//    std::sort(temp_arr, temp_arr + 8);
//    for (int j = 0; j < 8; ++j) {
//      intermediate_arr[k * 8 + j] = temp_arr[j];
//      check_arr[k * 8 + j] = temp_arr[j];
//    }
//  }
//
//  for (int l = 0; l < 4; ++l) {
//    std::sort(check_arr + l * 16, check_arr + l * 16 + 16);
//  }
//
//  int *buffer;
//  aligned_init(buffer, 64);
//  AVX256MergeUtil::MergePass8<int,__m512i>(intermediate_arr, buffer, 64, 8);
//
//  for (int m = 0; m < 64; ++m) {
//    EXPECT_EQ(check_arr[m], buffer[m]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  free(check_arr);
//  free(temp_arr);
//  free(buffer);
//}
//
//TEST(MergeUtilsTest, AVX256MergePass8Float32BitTest) {
//  float *arr, *intermediate_arr;
//  aligned_init<float>(arr, 64);
//  aligned_init<float>(intermediate_arr, 64);
//
//  TestUtil::RandGenFloat<float>(arr, 64, -10, 10);
//
//  auto *check_arr = (float *)malloc(64 * sizeof(float));
//  auto *temp_arr = (float *)malloc(8 * sizeof(float));
//
//  for (int k = 0; k < 8; k++) {
//    for (int i = 0; i < 8; ++i) {
//      temp_arr[i] = arr[i * 8 + k];
//    }
//    std::sort(temp_arr, temp_arr + 8);
//    for (int j = 0; j < 8; ++j) {
//      intermediate_arr[k * 8 + j] = temp_arr[j];
//      check_arr[k * 8 + j] = temp_arr[j];
//>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
//    }
//  }
//
//  for (int l = 0; l < 4; ++l) {
//    std::sort(check_arr + l * 16, check_arr + l * 16 + 16);
//  }
//
//  float *buffer;
//  aligned_init<float>(buffer, 64);
//  AVX256MergeUtil::MergePass8<float,__m512>(intermediate_arr, buffer, 64, 8);
//
//  for (int m = 0; m < 64; ++m) {
//    EXPECT_EQ(check_arr[m], buffer[m]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  delete[](buffer);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergePass4Int64BitTest) {
//  int64_t *arr, *intermediate_arr;
//  aligned_init<int64_t>(arr, 16);
//  aligned_init<int64_t>(intermediate_arr, 16);
//
//  TestUtil::RandGenInt<int64_t>(arr, 16, -10, 10);
//
//  auto *check_arr = (int64_t *)malloc(16 * sizeof(int64_t));
//  auto *temp_arr = (int64_t *)malloc(4 * sizeof(int64_t));
//
//  for (int k = 0; k < 4; k++) {
//    for (int i = 0; i < 4; ++i) {
//      temp_arr[i] = arr[i * 4 + k];
//    }
//    std::sort(temp_arr, temp_arr + 4);
//    for (int j = 0; j < 4; ++j) {
//      intermediate_arr[k * 4 + j] = temp_arr[j];
//      check_arr[k * 4 + j] = temp_arr[j];
//    }
//  }
//
//  for (int l = 0; l < 2; ++l) {
//    std::sort(check_arr + l * 8, check_arr + l * 8 + 8);
//  }
//
//  int64_t *buffer;
//  aligned_init<int64_t>(buffer, 16);
//  AVX256MergeUtil::MergePass8<int64_t,__m512>(intermediate_arr, buffer, 16, 4);
//
//  for (int m = 0; m < 16; ++m) {
//    EXPECT_EQ(check_arr[m], buffer[m]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  delete[](buffer);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256MergePass4Float64BitTest) {
//  double *arr, *intermediate_arr;
//  aligned_init<double>(arr, 16);
//  aligned_init<double>(intermediate_arr, 16);
//
//  TestUtil::RandGenInt<double>(arr, 16, -10, 10);
//
//  auto *check_arr = (double *)malloc(16 * sizeof(double));
//  auto *temp_arr = (double *)malloc(4 * sizeof(double));
//
//  for (int k = 0; k < 4; k++) {
//    for (int i = 0; i < 4; ++i) {
//      temp_arr[i] = arr[i * 4 + k];
//    }
//    std::sort(temp_arr, temp_arr + 4);
//    for (int j = 0; j < 4; ++j) {
//      intermediate_arr[k * 4 + j] = temp_arr[j];
//      check_arr[k * 4 + j] = temp_arr[j];
//    }
//  }
//
//  for (int l = 0; l < 2; ++l) {
//    std::sort(check_arr + l * 8, check_arr + l * 8 + 8);
//  }
//
//  double *buffer;
//  aligned_init<double>(buffer, 16);
//  AVX256MergeUtil::MergePass8<double,__m512d>(intermediate_arr, buffer, 16, 4);
//
//  for (int m = 0; m < 16; ++m) {
//    EXPECT_EQ(check_arr[m], buffer[m]);
//  }
//
//  delete[](arr);
//  delete[](intermediate_arr);
//  delete[](buffer);
//  free(check_arr);
//  free(temp_arr);
//}
//
//TEST(MergeUtilsTest, AVX256IntraRegisterSort8x8Int32BitTest) {
//  int *a;
//  int *b;
//  aligned_init<int>(a, 8);
//  aligned_init<int>(b, 8);
//
//  TestUtil::RandGenInt<int>(a, 8, -10, 10);
//  TestUtil::RandGenInt<int>(b, 8, -10, 10);
//
//  std::sort(a, a + 8);
//  std::sort(b, b + 8);
//
//  int check_arr[16];
//  for (int i = 0; i < 16; ++i) {
//    check_arr[i] = i < 8 ? a[i] : b[i - 8];
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  std::reverse(b, b + 8);
//
//  __m256i ra, rb;
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::IntraRegisterSort8x8(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for (int j = 0; j < 16; ++j) {
//    EXPECT_EQ(check_arr[j], j < 8 ? a[j] : b[j - 8]);
//  }
//
//  delete[](a);
//  delete[](b);
//}
//<<<<<<< HEAD
//}
//=======
//
//TEST(MergeUtilsTest, AVX256IntraRegisterSort8x8Float32BitTest) {
//  float *a;
//  float *b;
//  aligned_init<float>(a, 8);
//  aligned_init<float>(b, 8);
//
//  TestUtil::RandGenFloat<float>(a, 8, -10, 10);
//  TestUtil::RandGenFloat<float>(b, 8, -10, 10);
//
//  std::sort(a, a + 8);
//  std::sort(b, b + 8);
//
//  float check_arr[16];
//  for (int i = 0; i < 16; ++i) {
//    check_arr[i] = i < 8 ? a[i] : b[i - 8];
//  }
//
//  std::sort(check_arr, check_arr + 16);
//
//  std::reverse(b, b + 8);
//
//  __m256 ra, rb;
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::IntraRegisterSort8x8(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for (int j = 0; j < 16; ++j) {
//    EXPECT_EQ(check_arr[j], j < 8 ? a[j] : b[j - 8]);
//  }
//
//  delete[](a);
//  delete[](b);
//}
//
//TEST(MergeUtilsTest, AVX256IntraRegisterSort4x4Int64BitTest) {
//  int64_t *a;
//  int64_t *b;
//  aligned_init<int64_t>(a, 4);
//  aligned_init<int64_t>(b, 4);
//
//  TestUtil::RandGenInt<int64_t>(a, 4, -10, 10);
//  TestUtil::RandGenInt<int64_t>(b, 4, -10, 10);
//
//  std::sort(a, a + 4);
//  std::sort(b, b + 4);
//
//  int64_t check_arr[8];
//  for (int i = 0; i < 8; ++i) {
//    check_arr[i] = i < 4 ? a[i] : b[i - 4];
//  }
//
//  std::sort(check_arr, check_arr + 8);
//
//  std::reverse(b, b + 4);
//
//  __m256i ra, rb;
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::IntraRegisterSort4x4(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for (int j = 0; j < 8; ++j) {
//    EXPECT_EQ(check_arr[j], j < 4 ? a[j] : b[j - 4]);
//  }
//
//  delete[](a);
//  delete[](b);
//}
//
//TEST(MergeUtilsTest, AVX256IntraRegisterSort4x4Float64BitTest) {
//  double *a;
//  double *b;
//  aligned_init<double>(a, 4);
//  aligned_init<double>(b, 4);
//
//  TestUtil::RandGenFloat<double>(a, 4, -10, 10);
//  TestUtil::RandGenFloat<double>(b, 4, -10, 10);
//
//  std::sort(a, a + 4);
//  std::sort(b, b + 4);
//
//  double check_arr[8];
//  for (int i = 0; i < 8; ++i) {
//    check_arr[i] = i < 4 ? a[i] : b[i - 4];
//  }
//
//  std::sort(check_arr, check_arr + 8);
//
//  std::reverse(b, b + 4);
//
//  __m256d ra, rb;
//  AVX256Util::LoadReg(ra, a);
//  AVX256Util::LoadReg(rb, b);
//  AVX256MergeUtil::IntraRegisterSort4x4(ra, rb);
//  AVX256Util::StoreReg(ra, a);
//  AVX256Util::StoreReg(rb, b);
//
//  for (int j = 0; j < 8; ++j) {
//    EXPECT_EQ(check_arr[j], j < 4 ? a[j] : b[j - 4]);
//  }
//
//  delete[](a);
//  delete[](b);
//}
//
//>>>>>>> d1d42f31014ccc9ad722d365f92651609e7a68b9
