#pragma once

#include "common.h"
#include <random>

struct TestUtil{
  template <typename T>
  static void PopulateSeqArray(T *&arr, int start, int end, int step=1) {
    int idx = 0;
    for(int i = start; i < end; i+=step) {
      arr[idx++] = i;
    }
  }

  template <typename T>
  static void RandGen(T* &arr, int N, T lo, T hi) {
    aligned_init<T>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i] = dis(gen);
    }
  }

  static void RandPairGen(std::pair<int,int>* &arr, int N, int lo, int hi) {
    aligned_init<std::pair<int,int>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i] = std::pair<int,int>(dis(gen), dis(gen));
    }
  }
};