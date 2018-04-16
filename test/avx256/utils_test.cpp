#include "gtest/gtest.h"
#include "common.h"
#include "avx256/utils.h"

TEST(UtilsTest, MinMax8Test) {
  int a[8];
  int b[8];
  __m256i ra, rb;
  for(int i = 0; i < 16; i++) {
    if(i % 2 == 0) {
      b[i/2] = i;
    } else {
      a[i/2] = i;
    }
  }
  for(int i = 0; i < 8; i++) {
    EXPECT_GE(a[i], b[i]);
  }
  AVX256Util::LoadReg(ra, a);
  AVX256Util::LoadReg(rb, b);
  AVX256Util::MinMax8(ra, rb);
  AVX256Util::StoreReg(ra, a);
  AVX256Util::StoreReg(rb, b);
  for(int i = 0; i < 8; i++) {
    EXPECT_LE(a[i], b[i]);
  }
}