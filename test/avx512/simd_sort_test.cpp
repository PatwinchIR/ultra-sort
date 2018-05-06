#include "metrics/cycletimer.h"
#include "gtest/gtest.h"
#include "test_util.h"
#include "avx512/simd_sort.h"
#include <algorithm>
#include <iterator>
#include "ips4o.hpp"
#include "pdqsort.h"

#ifdef AVX512

namespace avx512 {
TEST(SIMDSortTests, AVX512SIMDSort32BitIntegerTest) {
  size_t N = NNUM;
  int lo = LO;
  int hi = HI;
  int *rand_arr;
  int *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenInt(rand_arr, N, lo, hi);

  // C++ std::stable_sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ ips4o::sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ pqd::sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 Sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<int> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  // First perform a correctness check
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort32BitFloatTest) {
  size_t N = NNUM;
  float lo = LO;
  float hi = HI;
  float *rand_arr;
  float *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenFloat<float>(rand_arr, N, lo, hi);

  // C++ std::stable_sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ ips4o::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ pqd::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 Sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<float> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  // First perform a correctness check
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort64BitIntegerTest) {
  size_t N = NNUM;
  int lo = LO;
  int hi = HI;
  int64_t *rand_arr;
  int64_t *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenInt<int64_t>(rand_arr, N, lo, hi);

  // C++ std::stable_sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ ips4o::sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ pqd::sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init<int64_t>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<int64_t> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  for(unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort64BitFloatTest) {
  size_t N = NNUM;
  double lo = LO;
  double hi = HI;
  double *rand_arr;
  double *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenFloat<double>(rand_arr, N, lo, hi);

  // C++ std::stable_sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ ips4o::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ pqd::sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N);
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<double> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end());
  for(unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i], soln_arr[i]);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort32BitKeyValueIntTest) {
  using T = int;
  size_t N = NNUM;
  T lo = LO;
  T hi = HI;
  std::pair<T, T> *rand_arr;
  std::pair<T, T> *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenIntRecords(rand_arr, N, lo, hi);
  std::map<T, T> kv_map;
  for (unsigned int i = 0; i < N; ++i) {
    kv_map.insert(std::pair<T, T>(rand_arr[i].second, rand_arr[i].first));
  }

  // C++ std::stable_sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // ips4o
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // pdqsort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<std::pair<T, T>> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i].first, soln_arr[i].first);
    EXPECT_EQ(kv_map[soln_arr[i].second], soln_arr[i].first);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX256SIMDOrderBy32BitIntTest) {
  using T = int;
  size_t N = NNUM;
  T lo = LO;
  T hi = HI;
  std::pair<T, T> *rand_arr;
  std::pair<T, T> *soln_arr1, *soln_arr2, *input_arr1, *input_arr2;
  double start, end;

  // Initialization
  TestUtil::RandGenIntEntries(rand_arr, N, lo, hi);

  aligned_init<std::pair<T, T>>(input_arr1, N);
  aligned_init<std::pair<T, T>>(soln_arr1, N);
  std::copy(rand_arr, rand_arr + N, input_arr1);
  std::vector<std::pair<T, T>> check_arr1(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDOrderBy(soln_arr1, N, input_arr1);
  end = currentSeconds();
  std::sort(check_arr1.begin(), check_arr1.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr1[i].first, soln_arr1[i].first);
  }

  aligned_init<std::pair<T, T>>(input_arr2, N);
  aligned_init<std::pair<T, T>>(soln_arr2, N);
  std::copy(rand_arr, rand_arr + N, input_arr2);
  std::vector<std::pair<T, T>> check_arr2(rand_arr, rand_arr + N);
  SIMDOrderBy(soln_arr2, N, input_arr2, 1);
  std::sort(check_arr2.begin(), check_arr2.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.second < right.second;
  });
  for (int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr2[i].second, soln_arr2[i].second);
  }
  printf("[avx256::orderby] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr1;
  delete soln_arr2;
}

TEST(SIMDSortTests, AVX512SIMDSort64BitKeyValueIntTest) {
  using T = int64_t;
  size_t N = NNUM;
  T lo = LO;
  T hi = HI;
  std::pair<T, T> *rand_arr;
  std::pair<T, T> *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenIntRecords(rand_arr, N, lo, hi);
  std::map<T, T> kv_map;
  for (unsigned int i = 0; i < N; ++i) {
    kv_map.insert(std::pair<T, T>(rand_arr[i].second, rand_arr[i].first));
  }

  // C++ std::stable_sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // ips4o::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // pdqsort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<std::pair<T, T>> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i].first, soln_arr[i].first);
    EXPECT_EQ(kv_map[soln_arr[i].second], soln_arr[i].first);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort32BitKeyValueFloatTest) {
  using T = float;
  size_t N = NNUM;
  T lo = LO;
  T hi = HI;
  std::pair<T, T> *rand_arr;
  std::pair<T, T> *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenFloatRecords(rand_arr, N, lo, hi);
  std::map<T, T> kv_map;
  for (unsigned int i = 0; i < N; ++i) {
    kv_map.insert(std::pair<T, T>(rand_arr[i].second, rand_arr[i].first));
  }

  // C++ std::stable_sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // ips4o::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // pdqsort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<std::pair<T, T>> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i].first, soln_arr[i].first);
    EXPECT_EQ(kv_map[soln_arr[i].second], soln_arr[i].first);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

TEST(SIMDSortTests, AVX512SIMDSort64BitKeyValueFloatTest) {
  using T = double;
  size_t N = NNUM;
  T lo = LO;
  T hi = HI;
  std::pair<T, T> *rand_arr;
  std::pair<T, T> *soln_arr;
  double start, end;

  // Initialization
  TestUtil::RandGenFloatRecords(rand_arr, N, lo, hi);
  std::map<T, T> kv_map;
  for (unsigned int i = 0; i < N; ++i) {
    kv_map.insert(std::pair<T, T>(rand_arr[i].second, rand_arr[i].first));
  }

  // C++ std::stable_sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::stable_sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++ std::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[std::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // ips4o::sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  ips4o::sort(soln_arr, soln_arr + N, [](const std::pair<T, T> &left, const std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[ips4o::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // pdqsort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  pdqsort(soln_arr, soln_arr + N, [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  end = currentSeconds();
  printf("[pdqsort] %lu elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // AVX512 sort
  aligned_init<std::pair<T, T>>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  std::vector<std::pair<T, T>> check_arr(rand_arr, rand_arr + N);
  start = currentSeconds();
  SIMDSort(N, soln_arr);
  end = currentSeconds();
  std::sort(check_arr.begin(), check_arr.end(), [](std::pair<T, T> &left, std::pair<T, T> &right) {
    return left.first < right.first;
  });
  for (unsigned int i = 0; i < N; i++) {
    EXPECT_EQ(check_arr[i].first, soln_arr[i].first);
    EXPECT_EQ(kv_map[soln_arr[i].second], soln_arr[i].first);
  }
  printf("[avx512::sort] %lu elements: %.8f seconds\n", N, end - start);
  delete rand_arr;
  delete soln_arr;
}

}

#endif