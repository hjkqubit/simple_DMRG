cmake_minimum_required(VERSION 3.19)

project(simple_dmrg)

# Find CUDA, BLAS, LAPACK, and CUDAToolKit
find_package(CUDA REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# Set cuTENSOR root
# Users should edit this link to their own cuTENSOR root location
set(CUTENSOR_ROOT "/usr/lib/x86_64-linux-gnu/libcutensor")

# Check cuTENSOR root
if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

# Add the cuTENSOR library path
link_directories(${CUTENSOR_ROOT}/12)

# Set default paths for xtl, xtensor, and xtensor-blas
# Users should edit this block to their own xtensor-related directory paths
set(xtl_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtl/include")
set(xtensor_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtensor/include")
set(xtensor_blas_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtensor-blas/include")

# Include the xtl-related directories
include_directories(${xtl_INCLUDE_DIR})
include_directories(${xtensor_INCLUDE_DIR})
include_directories(${xtensor_blas_INCLUDE_DIR})

# Set optimization flag
set(CMAKE_CXX_FLAG "-O3")

# Add subdirectories for CMake
add_subdirectory(examples)
