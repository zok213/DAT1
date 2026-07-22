cmake_minimum_required(VERSION 3.18)
project(cuda_bcs_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89) # T4, A100, RTX 30/40, Orin

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

add_library(cuda_kernels_lib STATIC
    src/cuda_kernels.cu
)

add_executable(cuda_bcs_pipeline
    src/main.cpp
    src/tensorrt_pipeline.cpp
)

target_link_libraries(cuda_bcs_pipeline
    PRIVATE
    cuda_kernels_lib
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)
