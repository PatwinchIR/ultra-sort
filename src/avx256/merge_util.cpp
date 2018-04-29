#include "avx256/merge_util.h"

#ifdef AVX2

template <typename InType, typename RegType>
void AVX256MergeUtil::MergeRuns8(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=8;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass8<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}
template void AVX256MergeUtil::MergeRuns8<int,__m256i>(int *&arr, int N);
template void AVX256MergeUtil::MergeRuns8<float,__m256>(float *&arr, int N);

template <typename InType, typename RegType>
void AVX256MergeUtil::MergeRuns4(InType *&arr, int N) {
  InType* buffer;
  int UNIT_RUN_SIZE=4;
  aligned_init(buffer, N);
  for(int run_size = UNIT_RUN_SIZE; run_size < N; run_size*=2) {
    MergePass4<InType,RegType>(arr, buffer, N, run_size);
    std::swap(arr, buffer);
  }
}

template void AVX256MergeUtil::MergeRuns4<int64_t,__m256i>(int64_t *&arr, int N);
template void AVX256MergeUtil::MergeRuns4<double,__m256d>(double *&arr, int N);

template <typename InType, typename RegType>
void AVX256MergeUtil::MergePass8(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=8;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX256Util::LoadReg(ra, &arr[p1_ptr]);
    AVX256Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge8(ra, rb);

      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX256Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX256Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge8(ra, rb);

    AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX256Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX256Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge8(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX256Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX256MergeUtil::MergePass8<int,__m256i>(int *&arr, int *buffer, int N, int run_size);
template void AVX256MergeUtil::MergePass8<float,__m256>(float *&arr, float *buffer, int N, int run_size);

template <typename InType, typename RegType>
void AVX256MergeUtil::MergePass4(InType *&arr, InType *buffer, int N, int run_size) {
  int UNIT_RUN_SIZE=4;
  RegType ra, rb;
  int buffer_offset = 0;
  for(int i = 0; i < N; i += 2*run_size) {
    int start = i;
    int mid = i + run_size;
    int end = i + 2*run_size;
    int p1_ptr = start;
    int p2_ptr = mid;
    AVX256Util::LoadReg(ra, &arr[p1_ptr]);
    AVX256Util::LoadReg(rb, &arr[p2_ptr]);
    p1_ptr += UNIT_RUN_SIZE;
    p2_ptr += UNIT_RUN_SIZE;

    while (p1_ptr < mid && p2_ptr < end) {
      BitonicMerge4(ra, rb);

      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;

      if(arr[p1_ptr] > arr[p2_ptr]) {
        AVX256Util::LoadReg(ra, &arr[p2_ptr]);
        p2_ptr += UNIT_RUN_SIZE;
      } else {
        AVX256Util::LoadReg(ra, &arr[p1_ptr]);
        p1_ptr += UNIT_RUN_SIZE;
      }
    }

    BitonicMerge4(ra, rb);

    AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;

    while (p1_ptr < mid) {
      AVX256Util::LoadReg(ra, &arr[p1_ptr]);
      p1_ptr += UNIT_RUN_SIZE;
      BitonicMerge4(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    while (p2_ptr < end) {
      AVX256Util::LoadReg(ra, &arr[p2_ptr]);
      p2_ptr += UNIT_RUN_SIZE;
      BitonicMerge4(ra, rb);
      AVX256Util::StoreReg(ra, &buffer[buffer_offset]);
      buffer_offset += UNIT_RUN_SIZE;
    }

    AVX256Util::StoreReg(rb, &buffer[buffer_offset]);
    buffer_offset += UNIT_RUN_SIZE;
  }
}

template void AVX256MergeUtil::MergePass4<int64_t,__m256i>(int64_t *&arr, int64_t *buffer, int N, int run_size);
template void AVX256MergeUtil::MergePass4<double,__m256d>(double *&arr, double *buffer, int N, int run_size);

void AVX256MergeUtil::IntraRegisterSort8x8(__m256i &a8, __m256i &b8) {
  __m256i mina, maxa, minb, maxb;
  // phase 1
  AVX256Util::MinMax8(a8, b8);
  auto a8_1 = _mm256_permutevar8x32_epi32(a8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
  auto b8_1 = _mm256_permutevar8x32_epi32(b8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));

  AVX256Util::MinMax8(a8, a8_1, mina, maxa);
  AVX256Util::MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm256_blend_epi32(mina, maxa, 0xf0);
  auto b4 = _mm256_blend_epi32(minb, maxb, 0xf0);

  // phase 2
  auto a4_1 = _mm256_shuffle_epi32(a4, 0x4e);
  auto b4_1 = _mm256_shuffle_epi32(b4, 0x4e);

  AVX256Util::MinMax8(a4, a4_1, mina, maxa);
  AVX256Util::MinMax8(b4, b4_1, minb, maxb);

  auto a2 = _mm256_unpacklo_epi64(mina, maxa);
  auto b2 = _mm256_unpacklo_epi64(minb, maxb);

  // phase 3
  auto a2_1 = _mm256_shuffle_epi32(a2, 0xb1);
  auto b2_1 = _mm256_shuffle_epi32(b2, 0xb1);

  AVX256Util::MinMax8(a2, a2_1, mina, maxa);
  AVX256Util::MinMax8(b2, b2_1, minb, maxb);

  a8 = _mm256_blend_epi32(mina, maxa, 0xaa);
  b8 = _mm256_blend_epi32(minb, maxb, 0xaa);
}

void AVX256MergeUtil::IntraRegisterSort8x8(__m256 &a8, __m256 &b8) {
  __m256 mina, maxa, minb, maxb;
  // phase 1
  AVX256Util::MinMax8(a8, b8);
  auto a8_1 = _mm256_permutevar8x32_ps(a8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));
  auto b8_1 = _mm256_permutevar8x32_ps(b8, _mm256_set_epi32(3, 2, 1, 0, 7, 6, 5, 4));

  AVX256Util::MinMax8(a8, a8_1, mina, maxa);
  AVX256Util::MinMax8(b8, b8_1, minb, maxb);

  auto a4 = _mm256_blend_ps(mina, maxa, 0xf0);
  auto b4 = _mm256_blend_ps(minb, maxb, 0xf0);

  // phase 2
  auto a4_1 = _mm256_shuffle_ps(a4, a4, 0x4e);
  auto b4_1 = _mm256_shuffle_ps(b4, b4, 0x4e);

  AVX256Util::MinMax8(a4, a4_1, mina, maxa);
  AVX256Util::MinMax8(b4, b4_1, minb, maxb);

  auto a2 = (__m256)_mm256_unpacklo_pd((__m256d)mina, (__m256d)maxa);
  auto b2 = (__m256)_mm256_unpacklo_pd((__m256d)minb, (__m256d)maxb);

  // phase 3
  auto a2_1 = _mm256_shuffle_ps(a2, a2, 0xb1);
  auto b2_1 = _mm256_shuffle_ps(b2, b2, 0xb1);

  AVX256Util::MinMax8(a2, a2_1, mina, maxa);
  AVX256Util::MinMax8(b2, b2_1, minb, maxb);

  a8 = _mm256_blend_ps(mina, maxa, 0xaa);
  b8 = _mm256_blend_ps(minb, maxb, 0xaa);
}

void AVX256MergeUtil::IntraRegisterSort4x4(__m256i &a4, __m256i &b4) {
  // Level 1
  AVX256Util::MinMax4(a4, b4);
  auto l1p = (__m256i)_mm256_permute2f128_pd((__m256d)a4, (__m256d)b4, 0x31);
  auto h1p = (__m256i)_mm256_permute2f128_pd((__m256d)a4, (__m256d)b4, 0x20);

  // Level 2
  AVX256Util::MinMax4(l1p, h1p);
  auto l2p = (__m256i)_mm256_shuffle_pd((__m256d)l1p, (__m256d)h1p, 0x0);
  auto h2p = (__m256i)_mm256_shuffle_pd((__m256d)l1p, (__m256d)h1p, 0xF);

  // Level 3
  AVX256Util::MinMax4(l2p, h2p);
  auto l3p = (__m256i)_mm256_unpacklo_pd((__m256d)l2p, (__m256d)h2p);
  auto h3p = (__m256i)_mm256_unpackhi_pd((__m256d)l2p, (__m256d)h2p);

  // Finally
  a4 = (__m256i)_mm256_permute2f128_pd((__m256d)l3p, (__m256d)h3p, 0x20);
  b4 = (__m256i)_mm256_permute2f128_pd((__m256d)l3p, (__m256d)h3p, 0x31);
}

void AVX256MergeUtil::IntraRegisterSort4x4(__m256d &a4, __m256d &b4) {
  // Level 1
  AVX256Util::MinMax4(a4, b4);
  auto l1p = _mm256_permute2f128_pd(a4, b4, 0x31);
  auto h1p = _mm256_permute2f128_pd(a4, b4, 0x20);

  // Level 2
  AVX256Util::MinMax4(l1p, h1p);
  auto l2p = _mm256_shuffle_pd(l1p, h1p, 0x0);
  auto h2p = _mm256_shuffle_pd(l1p, h1p, 0xF);

  // Level 3
  AVX256Util::MinMax4(l2p, h2p);
  auto l3p = _mm256_unpacklo_pd(l2p, h2p);
  auto h3p = _mm256_unpackhi_pd(l2p, h2p);

  // Finally
  a4 = _mm256_permute2f128_pd(l3p, h3p, 0x20);
  b4 = _mm256_permute2f128_pd(l3p, h3p, 0x31);
}

template <typename T>
void AVX256MergeUtil::BitonicMerge8(T &a, T &b) {
  b = AVX256Util::Reverse8(b);
  IntraRegisterSort8x8(a,b);
}

template void AVX256MergeUtil::BitonicMerge8<__m256i>(__m256i &a, __m256i &b);
template void AVX256MergeUtil::BitonicMerge8<__m256>(__m256 &a, __m256 &b);

template <typename T>
void AVX256MergeUtil::BitonicMerge4(T &a, T &b) {
  b = AVX256Util::Reverse4(b);
  IntraRegisterSort4x4(a,b);
}

template void AVX256MergeUtil::BitonicMerge4<__m256i>(__m256i &a, __m256i &b);
template void AVX256MergeUtil::BitonicMerge4<__m256d>(__m256d &a, __m256d &b);

#endif
