#include "metrics/cycletimer.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx256/simd_sort.h"
#include <algorithm>
#include <iterator>

TEST(SIMDSortTests, SIMDSort32BitIntegerTest) {
  int N = 65536;
  int lo = -1000;
  int hi = 1000;
  int *rand_arr;
  int *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGen(rand_arr, N, lo, hi);

  // C++11 std::stable_sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++11 std::sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<int> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSorter::SIMDSort32(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx256::sort] %d elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, SIMDSort64BitIntegerTest) {
  int N = 65536;
  int lo = -10000;
  int hi = 10000;
  int64_t *rand_arr;
  int64_t *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGen<int64_t>(rand_arr, N, lo, hi);

  // C++11 std::stable_sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++11 std::sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<int64_t> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSorter::SIMDSort64(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx256::sort] %d elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, SIMDSort32BitKeyValueIntegerTest) {
  int N = 65536;
  int lo = -10000;
  int hi = 10000;
  std::pair<int,int> *rand_arr;
  std::pair<int,int> *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandPairGen(rand_arr, N, lo, hi);

  // C++11 std::stable_sort
  aligned_init<std::pair<int,int>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N, [](const std::pair<int,int> &left, const std::pair<int,int> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::stable_sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++11 std::sort
  aligned_init<std::pair<int,int>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N, [](std::pair<int,int> &left, std::pair<int,int> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  aligned_init<std::pair<int,int>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<std::pair<int,int>> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSorter::SIMDSort32KV(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end(), [](std::pair<int,int> &left, std::pair<int,int> &right) {
    return left.first < right.first;
  });
  for(int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i].first, soln_arr[i].first);
  }
  printf("[avx256::sort] %d elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}