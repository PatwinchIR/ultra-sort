#include <iostream>
#include <cassert>
#include "metrics/cycletimer.h"
#include "common.h"
#include <random>
#include "sort/simd_sort.h"

void rand_gen(int* &arr, int N, int lo, int hi) {
  aligned_init<int>(arr, N);
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(lo, hi);
  for(size_t i = 0; i < N; i++) {
    arr[i] = dis(gen);
  }
}

void print_arr(int* arr, int N) {
  for(int i = 0; i < N; i++) {
    printf("%d, ", arr[i]);
  }
  printf("\n");
}


int main() {
  // Initialization
  int N = 64;
  int lo = -10;
  int hi = 10;
  int* rand_arr;
  int* soln_arr;
  double start, end;

  // Initialization
  rand_gen(rand_arr, N, lo, hi);


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


#ifdef __AVX__
#else
  printf("Missing AVX instructions\n");
#endif

#ifdef __AVX2__
  print_arr(rand_arr, N);
  sort_avx2(N, rand_arr);
  print_arr(rand_arr, N);
#else
  printf("Missing AVX2 instructions\n");
#endif

#ifdef __AVX512F__
#else
  printf("Missing AVX512 instructions\n");
#endif
  return 0;
}
