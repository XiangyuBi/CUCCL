include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
)

file(GLOB HEADERS 
CUCCL_LE/*.hpp CUCCL_LE/*.cuh CUCCL_LE/*.cu 
CUCCL_NP/*.h   CUCCL_NP/*.cuh CUCCL_NP/*.cu 
CUCCL_DPL/*.h  CUCCL_DPL/*.cuh CUCCL_DPL/*.cu
)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable(ccl_test 
${HEADERS}
test/test.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_35 -use_fast_math -std c++11
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
ccl_test
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
)

target_link_libraries( 
ccl_test
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
)

# FIND_PACKAGE( OpenCV QUIET)
# if(OpenCV_FOUND)
# 	include_directories(${OpenCV_INCLUDE_DIRS} )
# 	cuda_add_executable(ccl_opencv
# 		${HEADERS}
# 		test/test_cv.cu
# 		OPTIONS --cudart=shared -lineinfo -arch sm_35 -use_fast_math -std c++11
# 		DEBUG -Xptxas -v -g -G
# 		)

# 	target_link_libraries( 
# 		ccl_opencv
# 		${CUDA_CUDART_LIBRARY}
# 		${CUDA_CURAND_LIBRARY}
# 		${OpenCV_LIBS}
# 		)

# 	target_link_libraries( 
# 		ccl_opencv
# 		${CUDA_CUDART_LIBRARY}
# 		${CUDA_CURAND_LIBRARY}
# 		${OpenCV_LIBS}
# 	    )
# endif()


