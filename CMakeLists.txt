cmake_minimum_required(VERSION 3.5)
project(UltraSort)

set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set(CMAKE_CXX_STANDARD 14)

add_subdirectory(third_party/googletest)

include_directories(${CMAKE_SOURCE_DIR}/src/include)
include_directories(${CMAKE_SOURCE_DIR}/test/include)
include_directories(third_party/googletest/googletest/include)
include_directories(third_party/googletest/googlemock/include)
include_directories(third_party/ips4o)
include_directories(third_party/pdqsort)

set(CMAKE_CXX_FLAGS "-g -O3 -flto -Wall -march=native")

file(GLOB_RECURSE SOURCE_FILES
        "src/*.cpp" "test/*.cpp")

file(GLOB_RECURSE HEADER_FILES "src/include/*.h" "test/include/*.h")

add_executable(ultrasort ${SOURCE_FILES} ${HEADER_FILES})
target_link_libraries(ultrasort gtest gtest_main)
target_link_libraries(ultrasort gmock gmock_main)

# Read: https://stackoverflow.com/questions/28939652/how-to-detect-sse-sse2-avx-avx2-avx-512-avx-128-fma-kcvi-availability-at-compile
