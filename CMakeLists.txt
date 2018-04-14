cmake_minimum_required(VERSION 3.5)
project(UltraSort)

set(CMAKE_CXX_STANDARD 11)
include_directories(${CMAKE_SOURCE_DIR}/src/include)
set(CMAKE_CXX_FLAGS "-g -O3 -Wall -march=native -DDEBUG=0")

file(GLOB_RECURSE SOURCE_FILES
        "src/*.cpp")

file(GLOB_RECURSE HEADER_FILES "src/include/*.h")

add_executable(ultrasort ${SOURCE_FILES} ${HEADER_FILES} src/include/common.cpp)

# Read: https://stackoverflow.com/questions/28939652/how-to-detect-sse-sse2-avx-avx2-avx-512-avx-128-fma-kcvi-availability-at-compile