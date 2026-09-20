cmake_minimum_required(VERSION 3.20)
project(DES_GPUQueingModel VERSION 1.0 LANGUAGES C)

include(CTest)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_library(sim_core STATIC
    src/gamma_random.c
    src/sim_core.c
)
target_include_directories(sim_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
if(NOT MSVC)
    target_compile_options(sim_core PRIVATE -Wall -Wextra -Wpedantic)
endif()
target_link_libraries(sim_core PUBLIC m)

add_executable(DES_GPUQueingModel
    src/main.c
    src/sweep.c
)
target_link_libraries(DES_GPUQueingModel PRIVATE sim_core)

if(BUILD_TESTING)
    add_executable(test_sim_core tests/test_sim_core.c)
    target_link_libraries(test_sim_core PRIVATE sim_core)
    add_test(NAME sim_core_regression COMMAND test_sim_core)

    add_executable(test_distributions tests/test_distributions.c)
    target_link_libraries(test_distributions PRIVATE sim_core)
    add_test(NAME distributions_statistical COMMAND test_distributions)
endif()

add_executable(benchmark_main
    src/benchmark_main.c
    src/benchmark.c
)
target_link_libraries(benchmark_main PRIVATE sim_core)
