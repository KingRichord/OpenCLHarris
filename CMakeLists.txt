cmake_minimum_required(VERSION 3.16)
project(opencl_harris_test)

set(CMAKE_CXX_STANDARD 14)
include_directories("${PROJECT_SOURCE_DIR}/OpenCL") # 这里直接将路径，这里加入OpenCL之后，在代码中就不需要加入/OpenCL头信息了

find_package(OpenCL REQUIRED)
SET(OpenCV_DIR "/home/moi/APP/include/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
add_executable(opencl_harris_test main.cpp)
target_include_directories(opencl_harris_test PUBLIC ${OpenCL_INCLUDE_DIRS})
target_link_libraries(opencl_harris_test     ${OpenCL_LIBRARY}     ${OpenCV_LIBRARIES} )