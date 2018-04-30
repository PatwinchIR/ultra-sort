#include "avx512/merge_util.h"

#ifdef AVX512

template <typename InType, typename RegType>
void AVX512MergeUtil::MergeRuns16(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=16;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass16<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX512MergeUtil::MergeRuns16<int,__m512i>(int *&arr, int N);
template void AVX512MergeUtil::MergeRuns16<float,__m512>(float *&arr, int N);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergeRuns8(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=8;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass8<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX512MergeUtil::MergeRuns8<int64_t,__m512i>(int64_t *&arr, int N);
template void AVX512MergeUtil::MergeRuns8<double,__m512d>(double *&arr, int N);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergePass16(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=16;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX512Util::LoadReg(ra, &arr[p1_ptr]);
    AVX512Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge16(ra, rb);

      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX512Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX512Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge16(ra, rb);

    AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX512Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge16(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX512Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge16(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX512Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX512MergeUtil::MergePass16<int,__m512i>(int *&arr, int *buffer, int N, int run_size);
template void AVX512MergeUtil::MergePass16<float,__m512>(float *&arr, float *buffer, int N, int run_size);

template <typename InType, typename RegType>
void AVX512MergeUtil::MergePass8(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=8;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX512Util::LoadReg(ra, &arr[p1_ptr]);
    AVX512Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge8(ra, rb);

      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX512Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX512Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge8(ra, rb);

    AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX512Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX512Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      AVX512Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX512Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX512MergeUtil::MergePass8<int64_t,__m512i>(int64_t *&arr, int64_t *buffer, int N, int run_size);
template void AVX512MergeUtil::MergePass8<double,__m512d>(double *&arr, double *buffer, int N, int run_size);

void AVX512MergeUtil::IntraRegisterSort8x8(__m512i &a8, __m512i &b8) {
  __m512i mina, maxa, minb, maxb;
  // phase 1
  AVX512Util::MinMax8(a8, b8);
  auto a8_1 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), a8);
  auto b8_1 = _mm512_permutexvar_epi64(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), b8);

  AVX512Util::MinMax8(a8, a8_1, mina, maxa);
  AVX512Util::MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm512_permutex2var_epi64(mina, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxa);
  auto b4 = _mm512_permutex2var_epi64(minb, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxb);

  // TODO: Convert
  // phase 2
//  auto a4_1 = _mm512_shuffle_epi32(a4, 0x4e);
//  auto b4_1 = _mm512_shuffle_epi32(b4, 0x4e);
//
//  MinMax8(a4, a4_1, mina, maxa);
//  MinMax8(b4, b4_1, minb, maxb);
//
//  auto a2 = _mm512_unpacklo_epi64(mina, maxa);
//  auto b2 = _mm512_unpacklo_epi64(minb, maxb);
//
//  // phase 3
//  auto a2_1 = _mm512_shuffle_epi32(a2, 0xb1);
//  auto b2_1 = _mm512_shuffle_epi32(b2, 0xb1);
//
//  MinMax8(a2, a2_1, mina, maxa);
//  MinMax8(b2, b2_1, minb, maxb);
//
//  a8 = _mm512_blend_epi32(mina, maxa, 0xaa);
//  b8 = _mm512_blend_epi32(minb, maxb, 0xaa);
}

void AVX512MergeUtil::IntraRegisterSort8x8(__m512d &a8, __m512d &b8) {
  __m512d mina, maxa, minb, maxb;
  // phase 1
  AVX512Util::MinMax8(a8, b8);
  auto a8_1 = _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), a8);
  auto b8_1 = _mm512_permutexvar_pd(_mm512_set_epi64(3, 2, 1, 0, 7, 6, 5, 4), b8);

  AVX512Util::MinMax8(a8, a8_1, mina, maxa);
  AVX512Util::MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm512_permutex2var_pd(mina, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxa);
  auto b4 = _mm512_permutex2var_pd(minb, _mm512_set_epi64(8, 9, 10, 11, 4, 5, 6, 7), maxb);

  // TODO: Convert
  // phase 2
//  auto a4_1 = _mm512_shuffle_ps(a4, a4, 0x4e);
//  auto b4_1 = _mm512_shuffle_ps(b4, b4, 0x4e);
//
//  MinMax8(a4, a4_1, mina, maxa);
//  MinMax8(b4, b4_1, minb, maxb);
//
//  auto a2 = _mm512_unpacklo_pd(mina, maxa);
//  auto b2 = _mm512_unpacklo_pd(minb, maxb);
//
//  // phase 3
//  auto a2_1 = _mm512_shuffle_ps(a2, a2, 0xb1);
//  auto b2_1 = _mm512_shuffle_ps(b2, b2, 0xb1);
//
//  MinMax8(a2, a2_1, mina, maxa);
//  MinMax8(b2, b2_1, minb, maxb);
//
//  a8 = _mm512_blend_ps(mina, maxa, 0xaa);
//  b8 = _mm512_blend_ps(minb, maxb, 0xaa);
}

void AVX512MergeUtil::IntraRegisterSort16x16(__m512i& a16, __m512i& b16) {
  __m512i mina, maxa, minb, maxb;

  // phase 1
  AVX512Util::MinMax16(a16, b16);

  auto a16_1 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), a16);
  auto b16_1 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), b16);

  AVX512Util::MinMax16(a16, a16_1, mina, maxa);
  AVX512Util::MinMax16(b16, b16_1, minb, maxb);

  // phase 2
  auto a8 = _mm512_mask_blend_epi32((__mmask16)(0xff00), mina, maxa);
  auto b8 = _mm512_mask_blend_epi32((__mmask16)(0xff00), minb, maxb);

  auto a8_1 = _mm512_shuffle_i32x4(a8, a8, 0xb1);
  auto b8_1 = _mm512_shuffle_i32x4(b8, b8, 0xb1);

  AVX512Util::MinMax16(a8, a8_1, mina, maxa);
  AVX512Util::MinMax16(b8, b8_1, minb, maxb);

  // phase 3
  auto a4 = _mm512_mask_blend_epi32((__mmask16)(0xf0f0), mina, maxa);
  auto b4 = _mm512_mask_blend_epi32((__mmask16)(0xf0f0), minb, maxb);

  // https://clang.llvm.org/doxygen/avx512fintrin_8h_source.html
  auto a4_1 = _mm512_shuffle_epi32(a4, _MM_PERM_BADC); // 0x4e
  auto b4_1 = _mm512_shuffle_epi32(b4, _MM_PERM_BADC); // 0x4e

  AVX512Util::MinMax16(a4, a4_1, mina, maxa);
  AVX512Util::MinMax16(b4, b4_1, minb, maxb);

  // phase 4
  auto a2 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), mina, maxa);
  auto b2 = _mm512_mask_blend_epi32((__mmask16)(0xcccc), minb, maxb);

  auto a2_1 = _mm512_shuffle_epi32(a2, _MM_PERM_CDAB); // 0xb1
  auto b2_1 = _mm512_shuffle_epi32(b2, _MM_PERM_CDAB); // 0xb1

  AVX512Util::MinMax16(a2, a2_1, mina, maxa);
  AVX512Util::MinMax16(b2, b2_1, minb, maxb);

  a16 = _mm512_mask_blend_epi32((__mmask16)(0xaaaa), mina, maxa);
  b16 = _mm512_mask_blend_epi32((__mmask16)(0xaaaa), minb, maxb);
}

void AVX512MergeUtil::IntraRegisterSort16x16(__m512& a16, __m512& b16) {
  __m512 mina, maxa, minb, maxb;

  // phase 1
  AVX512Util::MinMax16(a16, b16);
  auto a16_1 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), a16);
  auto b16_1 = _mm512_permutexvar_ps(_mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8), b16);

  AVX512Util::MinMax16(a16, a16_1, mina, maxa);
  AVX512Util::MinMax16(b16, b16_1, minb, maxb);

  // phase 2
  auto a8 = _mm512_mask_blend_ps((__mmask16)(0xff00), mina, maxa);
  auto b8 = _mm512_mask_blend_ps((__mmask16)(0xff00), minb, maxb);

  auto a8_1 = _mm512_shuffle_f32x4(a8, a8, 0xb1);
  auto b8_1 = _mm512_shuffle_f32x4(b8, b8, 0xb1);

  AVX512Util::MinMax16(a8, a8_1, mina, maxa);
  AVX512Util::MinMax16(b8, b8_1, minb, maxb);

  // phase 3
  auto a4 = _mm512_mask_blend_ps((__mmask16)(0xf0f0), mina, maxa);
  auto b4 = _mm512_mask_blend_ps((__mmask16)(0xf0f0), minb, maxb);

  auto a4_1 = _mm512_shuffle_ps(a4, a4, 0x4e);
  auto b4_1 = _mm512_shuffle_ps(b4, b4, 0x4e);

  AVX512Util::MinMax16(a4, a4_1, mina, maxa);
  AVX512Util::MinMax16(b4, b4_1, minb, maxb);

  // phase 4
  auto a2 = _mm512_mask_blend_ps((__mmask16)(0xcccc), mina, maxa);
  auto b2 = _mm512_mask_blend_ps((__mmask16)(0xcccc), minb, maxb);

  auto a2_1 = _mm512_shuffle_ps(a2, a2, 0xb1);
  auto b2_1 = _mm512_shuffle_ps(b2, b2, 0xb1);

  AVX512Util::MinMax16(a2, a2_1, mina, maxa);
  AVX512Util::MinMax16(b2, b2_1, minb, maxb);

  a16 = _mm512_mask_blend_ps((__mmask16)(0xaaaa), mina, maxa);
  b16 = _mm512_mask_blend_ps((__mmask16)(0xaaaa), minb, maxb);
}

template <typename T>
void AVX512MergeUtil::BitonicMerge8(T& a, T& b) {
  b = Reverse8(b);
  IntraRegisterSort8x8(a,b);
}

template void AVX512MergeUtil::BitonicMerge8<__m512i>(__m512i &a, __m512i &b);
template void AVX512MergeUtil::BitonicMerge8<__m512d>(__m512d &a, __m512d &b);

template <typename T>
void AVX512MergeUtil::BitonicMerge16(T& a, T& b) {
  b = Reverse16(b);
  IntraRegisterSort16x16(a,b);
}

template void AVX512MergeUtil::BitonicMerge16<__m512i>(__m512i &a, __m512i &b);
template void AVX512MergeUtil::BitonicMerge16<__m512>(__m512 &a, __m512 &b);

#endif