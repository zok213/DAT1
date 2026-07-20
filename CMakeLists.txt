cmake_minimum_required(VERSION 3.10)
project(rk3588_cow_bcs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# In a real Radxa environment, these would point to the system libraries
# include_directories(/usr/include/rknn /usr/include/rga /usr/include/rockchip)
# link_directories(/usr/lib/aarch64-linux-gnu /usr/lib)

add_executable(rk3588_cow_bcs src/rk3588_pipeline.cpp)

# Link Rockchip proprietary libraries and standard threading
target_link_libraries(rk3588_cow_bcs
    pthread
    # rknnrt
    # rga
    # rockchip_mpp
)
