cmake_minimum_required(VERSION 3.18)
project(rm_armor_tensorrt LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS $ENV{TENSORRT_ROOT}/include /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS $ENV{TENSORRT_ROOT}/lib $ENV{TENSORRT_ROOT}/lib64 /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib)
if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found; set TENSORRT_ROOT to the TensorRT installation")
endif()

add_executable(rm_armor main.cpp)
target_include_directories(rm_armor PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(rm_armor PRIVATE ${OpenCV_LIBS} ${TENSORRT_LIBRARY} CUDA::cudart)
target_compile_options(rm_armor PRIVATE -O3 -Wall -Wextra)
