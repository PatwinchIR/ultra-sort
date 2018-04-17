#include <random>
#include "gtest/gtest.h"
#include "common.h"
#include "avx256/utils.h"

void PopulateSeqArray(int *&arr, int start, int end, int step=1) {
  int idx = 0;
  for(int i = start; i < end; i+=step) {
    arr[idx++] = i;
  }
}

template <typename T>
void rand_gen(T* &arr, int N, int lo, int hi) {
  aligned_init<T>(arr, N);
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(lo, hi);
  for(size_t i = 0; i < N; i++) {
    arr[i] = dis(gen);
  }
}

TEST(UtilsTest, LoadStoreTest) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);
  PopulateSeqArray(a, 0, 8);
  PopulateSeqArray(b, 8, 16);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::StoreReg(ra, b);
  AVX256Util::StoreReg(rb, a);
  for(int i = 0; i < 8; i++) {
    EXPECT_EQ(b[i], i);
    EXPECT_EQ(a[i], i + 8);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, MinMax8Test) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);
  PopulateSeqArray(a, 8, 16);
  PopulateSeqArray(b, 0, 8);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::MinMax8(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
  delete[](a);
  delete[](b);
}

TEST(UtilsTest, BitonicSort8x8Test) {
  int *arr;
  aligned_init<int>(arr, 64);
  rand_gen<int>(arr, 64, -10, 10);
  __m256i r[8];
  for(int i = 0; i < 8; i++) {
    AVX256Util::LoadReg(r[i], arr + i*8);
  }
  AVX256Util::BitonicSort8x8(r[0], r[1], r[2], r[3],
                             r[4], r[5], r[6], r[7]);
  for(int i = 0; i < 8; i++) {
    AVX256Util::StoreReg(r[i], arr + i*8);
  }
  for(int i = 8; i < 64; i+=8) {
    for(int j = i; j < i + 8; j++) {
      EXPECT_LE(arr[j-8], arr[j]);
    }
  }
  delete[](arr);
}

TEST(UtilsTest, BitonicMerge8) {
  int *a;
  int *b;
  aligned_init<int>(a, 8);
  aligned_init<int>(b, 8);
  PopulateSeqArray(a, 0, 16, 2);
  PopulateSeqArray(b, 1, 16, 2);
  __m256i ra, rb;
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::BitonicMerge8(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  int ab[16];
  for(int i = 0; i < 16; i++) {
    if(i < 8) {
      ab[i] = a[i];
    } else {
      ab[i] = b[i/2];
    }
    if(i > 1) {
      EXPECT_LE(ab[i - 1], ab[i]);
    }
  }
}
