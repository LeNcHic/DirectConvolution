cmake_minimum_required(VERSION 3.17)
project(elbrus_convolution)

set(CMAKE_CXX_STANDARD 11)

add_definitions(-DEIGEN_DONT_PARALLELIZE)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -fsanitize=address")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(mini_test mini_test.cpp)
target_link_libraries(mini_test Eigen3::Eigen)