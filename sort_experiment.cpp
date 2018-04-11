#include <cassert>
#include "metrics/cycletimer.h"
#include "common.h"
#include <random>
#include "sort/simd_sort_avx256.h"

void rand_gen(int* &arr, int N, int lo, int hi) {
  aligned_init<int>(arr, N);
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> dis(lo, hi);
  for(size_t i = 0; i < N; i++) {
    arr[i] = dis(gen);
  }
//  for(int i = 0; i < N; i++) {
//    arr[i] = N - i;
//  }
}

void print_arr(int* arr, int N) {
  for(int i = 0; i < N; i++) {
    printf("%d, ", arr[i]);
  }
  printf("\n");
}

void check_correctness(int* cand, int N) {
  for(int i = 1; i < N; i++) {
    assert(cand[i] >= cand[i - 1]);
  }
}


int main() {
  // Initialization
  // Need to resolve pass buffer problem for all multiples of 2
  int N = 128;
  int lo = -500;
  int hi = 500;
  int *rand_arr;
  int *soln_arr;
  double start, end;

  // Initialization
  rand_gen(rand_arr, N, lo, hi);


  // C++11 std::stable_sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::stable_sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  check_correctness(soln_arr, N);
  printf("[std::stable_sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

  // C++11 std::sort
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  std::sort(soln_arr, soln_arr + N);
  end = currentSeconds();
  check_correctness(soln_arr, N);
  printf("[std::sort] %d elements: %.8f seconds\n", N, end - start);
  delete soln_arr;

#ifdef __AVX__
#else
  printf("Missing AVX instructions\n");
#endif

#ifdef __AVX2__
  // TODO: Minor bug - we need to initialize and pass buffer here itself!!
  aligned_init<int>(soln_arr, N);
  std::copy(rand_arr, rand_arr + N, soln_arr);
  start = currentSeconds();
  sort_avx2(N, soln_arr);
  end = currentSeconds();
  print_arr(soln_arr, N);
  check_correctness(soln_arr, N);
  printf("[avx256::sort] %d elements: %.8f seconds\n", N, end - start);
#else
  printf("Missing AVX2 instructions\n");
#endif

#ifdef __AVX512F__
  sort_block_avx512(rand_arr, 0);
#else
  printf("Missing AVX512 instructions\n");
#endif
  return 0;
}
