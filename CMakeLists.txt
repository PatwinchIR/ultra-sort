cmake_minimum_required(VERSION 3.9)
project(UltraSort)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_C_STANDARD 11)

set(CMAKE_CXX_COMPILER g++)
set(SOURCE_FILES main.cpp saxpy/simd_saxpy.cpp metrics/cycletimer.cpp)
set(HEADER_FILES saxpy/simd_saxpy.h metrics/cycletimer.h)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -march=native -DDEBUG=0")
add_executable(ultrasort ${SOURCE_FILES} ${HEADER_FILES})