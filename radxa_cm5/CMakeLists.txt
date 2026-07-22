cmake_minimum_required(VERSION 3.14)
project(rk3588_cow_bcs LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    /usr/include/rknn
    /usr/include/mpp
    /usr/include/rga
)

set(SOURCES
    src/main.cpp
    src/rk3588_pipeline.cpp
    src/mpp_decoder.cpp
    src/rga_resizer.cpp
    src/rknn_engine.cpp
    src/bcs_classifier.cpp
)

add_executable(rk3588_cow_bcs ${SOURCES})

# Target linking for RK3588 hardware libraries when available
find_library(RKNN_RT_LIB rknn_api)
if(RKNN_RT_LIB)
    target_link_libraries(rk3588_cow_bcs PRIVATE ${RKNN_RT_LIB})
endif()

find_library(MPP_LIB mpp)
if(MPP_LIB)
    target_link_libraries(rk3588_cow_bcs PRIVATE ${MPP_LIB})
endif()

find_library(RGA_LIB rga)
if(RGA_LIB)
    target_link_libraries(rk3588_cow_bcs PRIVATE ${RGA_LIB})
endif()

target_link_libraries(rk3588_cow_bcs PRIVATE pthread)

message(STATUS "Configured Radxa CM5 (RK3588) Cow BCS target with modular C++ structure.")
