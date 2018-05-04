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
  static void RandGenInt(T* &arr, size_t N, T lo, T hi) {
    aligned_init<T>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i] = dis(gen);
    }
  }

  template <typename T>
  static void RandGenFloat(T* &arr, size_t N, T lo, T hi) {
    aligned_init<T>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i] = dis(gen);
    }
  }

  template <typename T>
  static void RandGenIntRecords(std::pair<T, T>* &arr, size_t N, T lo, T hi, int offset_start=0) {
    aligned_init<std::pair<T, T>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i].first = dis(gen);
      arr[i].second = offset_start++;
    }
  }

  template <typename T>
  static void RandGenIntRecords(T* &arr, size_t N, T lo, T hi, int offset_start=0) {
    aligned_init<T>(arr, N*2);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[2*i] = dis(gen);
      arr[2*i + 1] = offset_start++;
    }
  }

  template <typename T>
  static void RandGenFloatRecords(std::pair<T, T>* &arr, size_t N, T lo, T hi, int offset_start=0) {
    aligned_init<std::pair<T, T>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i].first = dis(gen);
      arr[i].second = offset_start++;
    }
  }

  template <typename T>
  static void RandGenFloatRecords(T* &arr, size_t N, T lo, T hi, int offset_start=0) {
    aligned_init<T>(arr, N*2);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[2*i] = dis(gen);
      arr[2*i + 1] = offset_start++;
    }
  }
};