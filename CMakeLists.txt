cmake_minimum_required(VERSION 3.19)

project(simple_dmrg)

find_package(CUDA REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(CUDAToolkit REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

# set cuTENSOR root
set(CUTENSOR_ROOT "/usr/lib/x86_64-linux-gnu/libcutensor")

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

link_directories(${CUTENSOR_ROOT}/12)

set(xtl_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtl/include")
set(xtensor_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtensor/include")
set(xtensor_blas_INCLUDE_DIR "/home/hjkqubit/Code/repos/xtensor-blas/include")
include_directories(${xtl_INCLUDE_DIR})
include_directories(${xtensor_INCLUDE_DIR})
include_directories(${xtensor_blas_INCLUDE_DIR})
