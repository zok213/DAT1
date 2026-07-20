cmake_minimum_required(VERSION 3.10)
project(COWdeployOptimizationSuite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Add OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    # Placeholders for ONNX, TFLite, and QNN paths
    /usr/include/onnxruntime
    /home/ubuntu/COWdeploy/qnn_sdk/qairt/2.48.0.260626/include/QNN
)

# Source files
set(SOURCES
    src/main.cpp
    src/onnx_runner.cpp
    src/tflite_runner.cpp
    src/qnn_runner.cpp
)

# Executable
add_executable(benchmark_runner ${SOURCES})

# Link libraries
target_link_libraries(benchmark_runner
    ${OpenCV_LIBS}
    # Placeholder for ONNX, TFLite, QNN libs
    # onnxruntime
    # tensorflow-lite
    # QnnHtp
)
