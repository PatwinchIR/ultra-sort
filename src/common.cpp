//
// Created by Dee Dong on 4/14/18.
//

#include "common.h"

template <typename T>
void aligned_init(T* &ptr, size_t N, size_t alignment_size) {
  if (posix_memalign((void **)&ptr, alignment_size, N*sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
}

template void aligned_init<int>(int* &ptr, size_t N, size_t alignment_size);
template void aligned_init<int64_t>(int64_t* &ptr, size_t N, size_t alignment_size);
template void aligned_init<double>(double* &ptr, size_t N, size_t alignment_size);
template void aligned_init<float>(float* &ptr, size_t N, size_t alignment_size);
template void aligned_init<std::pair<int,int>>(std::pair<int,int>* &ptr, size_t N, size_t alignment_size);
template void aligned_init<std::pair<int64_t,int64_t>>(std::pair<int64_t,int64_t>* &ptr, size_t N, size_t alignment_size);
template void aligned_init<std::pair<float,float>>(std::pair<float,float>* &ptr, size_t N, size_t alignment_size);
template void aligned_init<std::pair<double,double>>(std::pair<double,double>* &ptr, size_t N, size_t alignment_size);

template <typename T>
void print_arr(T *arr, int i, int j, const std::string &tag) {
  std::cout << tag.c_str() << std::endl;
  for(int idx = i; idx < j; idx++) {
    std::cout << arr[idx] << ", ";
  }
  std::cout << std::endl;
}
template void print_arr<int>(int* arr, int i, int j, const std::string &tag);
template void print_arr<int64_t>(int64_t* arr, int i, int j, const std::string &tag);
template void print_arr<double>(double* arr, int i, int j, const std::string &tag);
template void print_arr<float>(float* arr, int i, int j, const std::string &tag);

void print_kvarr(std::pair<int,int> *arr, int i, int j, const std::string &tag) {
  printf("%s ", tag.c_str());
  for(int idx = i; idx < j; idx++) {
    printf("(%d|%d), ", arr[idx].first, arr[idx].second);
  }
  printf("\n");
}

void print_kvarr(int64_t *arr, int i, int j, const std::string &tag) {
  printf("%s ", tag.c_str());
  for(int idx = i; idx < j; idx++) {
    int* arr_print = (int*)&arr[idx];
    printf("(%d|%d), ", arr_print[1], arr_print[0]);
  }
  printf("\n");
}