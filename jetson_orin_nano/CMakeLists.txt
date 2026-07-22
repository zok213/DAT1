cmake_minimum_required(VERSION 3.14)
project(jetson_cow_bcs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Path hints for Jetson CUDA and DeepStream SDKs
list(APPEND CMAKE_PREFIX_PATH "/usr/local/cuda" "/opt/nvidia/deepstream/deepstream")

find_package(CUDA QUIET)
find_package(OpenCV QUIET)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
    /usr/include/gstreamer-1.0
    /usr/include/glib-2.0
    /usr/lib/aarch64-linux-gnu/glib-2.0/include
    /opt/nvidia/deepstream/deepstream/sources/includes
)

set(SOURCES
    src/main.cpp
    src/jetson_pipeline.cpp
    src/nvmm_decoder.cpp
    src/tensorrt_engine.cpp
    src/bcs_classifier.cpp
)

add_executable(jetson_cow_bcs ${SOURCES})

# Conditional linking depending on hardware environment availability
if(CUDA_FOUND)
    target_link_libraries(jetson_cow_bcs PRIVATE ${CUDA_LIBRARIES})
endif()

if(OpenCV_FOUND)
    target_link_libraries(jetson_cow_bcs PRIVATE ${OpenCV_LIBS})
endif()

message(STATUS "Configured Jetson Cow BCS target with modular C++ structure.")
