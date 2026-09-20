cmake_minimum_required(VERSION 3.20)
project(DES_GPUQueingModel VERSION 1.0 LANGUAGES C)

include(CTest)
include(CheckLanguage)

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

check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_executable(cuda_gpu_benchmark src/cuda_gpu_benchmark.cu)
    set_target_properties(cuda_gpu_benchmark PROPERTIES
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
    )
    target_include_directories(cuda_gpu_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    target_link_libraries(cuda_gpu_benchmark PRIVATE sim_core
        CUDA::cudart
    )
    message(STATUS "CUDA benchmark enabled: cuda_gpu_benchmark")
else()
    message(STATUS "CUDA compiler not found; cuda_gpu_benchmark is disabled")
endif()
