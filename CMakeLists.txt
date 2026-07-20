cmake_minimum_required(VERSION 3.10)
project(jetson_cow_bcs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# In a real Jetson environment, these would point to the DeepStream/GStreamer libraries
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(GST REQUIRED gstreamer-1.0)
# include_directories(${GST_INCLUDE_DIRS} /opt/nvidia/deepstream/deepstream/sources/includes)
# link_directories(/opt/nvidia/deepstream/deepstream/lib)

add_executable(jetson_cow_bcs src/jetson_pipeline.cpp)

# Link NVIDIA proprietary libraries and GStreamer
target_link_libraries(jetson_cow_bcs
    pthread
    # nvdsgst_meta
    # nvds_meta
    # ${GST_LIBRARIES}
)
