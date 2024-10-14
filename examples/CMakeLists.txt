# Find uuid library
find_library(UUID_LIBRARIES NAMES uuid)
if(NOT UUID_LIBRARIES)
  message(FATAL_ERROR "Cannot find uuid library")
endif()

# Group all .cpp files from the src directory
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Compile dmrg_heisenberg
add_executable(dmrg_heisenberg dmrg_heisenberg.cpp ${SRC_FILES})

target_include_directories(dmrg_heisenberg PRIVATE ${CMAKE_SOURCE_DIR}/include/simple_DMRG)

target_link_libraries(dmrg_heisenberg PRIVATE LAPACK::LAPACK BLAS::BLAS ${UUID_LIBRARIES} ${CUTENSOR_ROOT}/12/libcutensor.so ${CUDA_LIBRARIES} ${CUDA_cublas_LIBRARY} CUDA::cusolver CUDA::cublas CUDA::cuda_driver)
