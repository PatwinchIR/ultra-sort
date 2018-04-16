//
// Created by Dee Dong on 4/14/18.
//

#include "common.h"

template <typename T>
void aligned_init(T* &ptr, int N, size_t alignment_size) {
  if (posix_memalign((void **)&ptr, alignment_size, N*sizeof(T)) != 0) {
    throw std::bad_alloc();
  }
}

template void aligned_init<int32_t>(int32_t* &ptr, int N, size_t alignment_size);
template void aligned_init<int64_t>(int64_t* &ptr, int N, size_t alignment_size);
template void aligned_init<std::pair<int,int>*>(std::pair<int,int>** &ptr, int N, size_t alignment_size);

void print_arr(int *arr, int i, int j, const std::string &tag) {
  printf("%s ", tag.c_str());
  for(int idx = i; idx < j; idx++) {
    printf("%d, ", arr[idx]);
  }
  printf("\n");
}

void printkv_arr(int64_t *arr, int i, int j, const std::string &tag) {
  printf("%s ", tag.c_str());
  for(int idx = i; idx < j; idx++) {
    int *arr_print = (int *)&arr[idx];
    printf("(%d|%d)", arr_print[1], arr_print[0]);
  }
  printf("\n");
}