include_directories(
  ${CUDA_TOOLKIT_INCLUDE}
)

file(GLOB HEADERS 
CUCCL_LE/*.hpp CUCCL_LE/*.cuh CUCCL_LE/*.cu 
CUCCL_NP/*.h   CUCCL_NP/*.cuh CUCCL_NP/*.cu 
)
source_group("Headers" FILES ${HEADERS})

cuda_add_executable(ccl 
${HEADERS}
test/test.cu
OPTIONS --cudart=shared  -lineinfo -arch sm_35 -use_fast_math -std c++11
DEBUG -Xptxas -v -g -G 
)

target_link_libraries( 
ccl
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
)

target_link_libraries( 
ccl
${CUDA_CUDART_LIBRARY}
${CUDA_CURAND_LIBRARY}
)
