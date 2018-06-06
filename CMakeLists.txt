cmake_minimum_required(VERSION 2.8)
project(ECE285GPUProgramming)

find_package(CUDA QUIET REQUIRED)
FIND_PACKAGE( OpenCV QUIET)

set (CMAKE_CXX_STANDARD 11)

file(GLOB HEADERS 
algo/CUCCL_LE/*.hpp algo/CUCCL_LE/*.cuh algo/CUCCL_LE/*.cu 
algo/CUCCL_NP/*.h   algo/CUCCL_NP/*.cuh algo/CUCCL_NP/*.cu 
algo/CUCCL_DPL/*.h  algo/CUCCL_DPL/*.cuh algo/CUCCL_DPL/*.cu
)
source_group("Headers" FILES ${HEADERS})


if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS} )
    cuda_add_executable(ccl_example
		${HEADERS}
		examples/kernel_evaluation.cpp
		OPTIONS --cudart=shared -lineinfo -arch sm_35 -use_fast_math -std c++11
		DEBUG -Xptxas -v -g -G
		)

    target_include_directories(ccl_example PRIVATE include )
    target_link_libraries( 
		ccl_example
		${CUDA_CUDART_LIBRARY}
		${CUDA_CURAND_LIBRARY}
		${OpenCV_LIBS}
		)

	target_link_libraries( 
		ccl_example
		${CUDA_CUDART_LIBRARY}
		${CUDA_CURAND_LIBRARY}
		${OpenCV_LIBS}
	    )
endif()

ADD_SUBDIRECTORY(algo)
