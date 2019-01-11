# Ultra-Sort
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
Extremely Parallelized SIMD Sorting Algorithm.
(All references are in the report.)

## Speedup achieved sorting 2^20 elements
Compared with `std::stable_sort`, `std::sort`, [`ips4o::sort`](https://github.com/SaschaWitt/ips4o), [`pdqsort`](https://github.com/orlp/pdqsort).

![](https://image.ibb.co/fnygES/analysis_AVX512_speedup.png)


# Usage:
1. Compile from `CMakelist.txt`
2. Due to integration with [`OpenMP`](http://www.openmp.org), run the executable after build using the command below:
```bash
export OMP_NUM_THREADS=1; ./ultrasort
```
By default, this will run all unit tests set up using [`GTest`](https://github.com/google/googletest).
To use this library in other projects simply include the header files:
```c++
#include "avx512/simd_sort.h"

avx512::SIMD_Sort(...); // avx256::... for AVX2 version.
```
The number of elements to sort is required to be a power of 2. More examples can be found at `test/avx512/simd_sort_test.cpp`.
