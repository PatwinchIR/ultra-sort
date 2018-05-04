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
  static void RandGenIntRecords(std::pair<T, T>* &arr, size_t N, T lo, T hi, unsigned int offset_start=0) {
    aligned_init<std::pair<T, T>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i].first = dis(gen);
      arr[i].second = offset_start++;
    }
  }

  template <typename T0, typename T1, typename T2, typename T3>
  static void RandGenMixedRecords(std::tuple<T0, T1, T2, T3>* &arr, size_t N, float lo, float hi) {
    aligned_init<std::tuple<T0, T1, T2, T3>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<float> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      // Casts everything from float to tuple specific type
      std::get<0>(arr[i]) = (T0)dis(gen);
      std::get<1>(arr[i]) = (T1)dis(gen);
      std::get<2>(arr[i]) = (T2)dis(gen);
      std::get<3>(arr[i]) = (T3)dis(gen);
    }
  }

  template <typename T>
  static void RandGenIntRecords(T* &arr, size_t N, T lo, T hi, unsigned int offset_start=0) {
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
  static void RandGenFloatRecords(std::pair<T, T>* &arr, size_t N, T lo, T hi, unsigned int offset_start=0) {
    aligned_init<std::pair<T, T>>(arr, N);
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<T> dis(lo, hi);
    for(size_t i = 0; i < N; i++) {
      arr[i].first = dis(gen);
      arr[i].second = offset_start++;
    }
  }

  template <typename T>
  static void RandGenFloatRecords(T* &arr, size_t N, T lo, T hi, unsigned int offset_start=0) {
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