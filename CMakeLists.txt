cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE "Release")

add_definitions("-Wall -g")

## CUDA
 find_package(CUDA PATHS /usr/local/cuda)
 include_directories(${CUDA_INCLUDE_DIRS})
##

set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_ARCHITECTURES 86)

project(ALIKE_cpp LANGUAGES CXX CUDA)
### OpenMP
find_package(OpenMP)
include_directories(${OpenMP_INCLUDE_DIRS})
###

### Torchlib
# set(Torch_DIR /home/)
find_package(Torch REQUIRED PATHS ${PROJECT_SOURCE_DIR}/libtorch)
message(STATUS "Found Torch include is ${TORCH_INCLUDE_DIRS}")
include_directories(${TORCH_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets")
###

set( DBoW3_INCLUDE_DIRS "/usr/local/include" )
set( DBoW3_LIBS "/usr/local/lib/libDBoW3.so" )

### OPENCV
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version is ${OpenCV_VERSION}")
message(STATUS "Found OpenCV include is ${OpenCV_INCLUDE_DIRS}")
include_directories(${OpenCV_INCLUDE_DIRS})
###

# the library that only has the cuda code
cuda_add_library(get_patches_cuda SHARED src/get_patches.cu)
set_target_properties(get_patches_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_library(get_patches SHARED src/get_patches.cpp)
target_link_libraries(get_patches get_patches_cuda)

include_directories(include)
add_executable(demo 
                src/main.cpp 
                src/utils.cpp)
target_link_libraries(demo ${TORCH_LIBRARIES} ${OpenCV_LIBS} get_patches)


add_executable(voc
        src/utils.cpp
        src/create_voc.cpp)
target_link_libraries(voc ${TORCH_LIBRARIES} ${OpenCV_LIBS} ${DBoW3_LIBS} get_patches)