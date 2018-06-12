

# CuCCL
## A Benchmark for CUDA Implementation of Connected Component Labeling  
### ECE 285 GPU Programming. 2018 Spring.

Implementing 3 GPU-accelerated CCL algorithms based on CUDA. Comparing performance with demonstrated CPU implementation. Validating and testing the result by intuitive visualization.

## Requirements


* CMake 3.0.0 or higher (https://cmake.org/download/)

* OpenCV 3.0 or higher (http://opencv.org/downloads.html),

* CUDA 8.0 or higher(https://developer.nvidia.com/cuda-downloads)

## Test Dataset 

* We use part of binary test images in  _YACCLAB_ Dataset to validate our implementation. For more details it can be inferred from http://imagelab.ing.unimore.it/yacclab .

![alt text](/dataset/im2200.png)![alt text](/dataset/im2201.png)


## Examples

CUCCL evaluates the performance of CUDA algorithm by comparing the elapsing time and the correctness by number of labels. In addition, we use OpenCV to colorize between adjacent labels. If can check any of these functionality by uncommenting the macros defined in _/example/kernel_evaluation.cpp_ .

    #define SAVEFILE 1        // Save the output files
    #define RUNTEST 1         // Perform the whole test
    #define RUNTIMETEST 1     // Perform timing check
    #define CORRECTNESSTEST 1 // Perform correctness check
    #define VISUALIZATION 1   // Visualization


Example for run Visualization :
    
    mkdir build
    cd build
    cmake ..
    make
    ./ccl_example LE ${PATH_OF_CUCCL}/dataset/ ${PATH_OF_CUCCL}/images/ ${IMAGE_NAME_TO_VISUALIZE}

Alternatively, to run more tests:

    ./ccl_example LE ${PATH_OF_CUCCL}/dataset/ ${PATH_OF_CUCCL}/images/ $(ls ../dataset)


## Algorithms 

* Kernel A – Neighbour Propagation

    A very simple multi-pass labelling method. It parallelises the task of labelling by creating one
thread for each cell which loads the field and label data from its cell and the neighbouring cells.
* Kernel B – Directional Propagation Labelling

    Kernel B is designed to overcome the problem that a label can only propagate itself by one cell per iteration (Kernel A) or one block per iteration

* Kernel C - Label Equivalence
   
    A multi-pass algorithm that records and resolves equivalences. 

For more details : [Parallel graph component labelling with GPUs and CUDA](https://www.sciencedirect.com/science/article/pii/S0167819110001055)


## Results

One can check the results by running the examples above. 

### Visualization
Some examples of visualization are shown below. Left side are origin images and right side are colorized ones.


![alt text](/images/result1.png)
![alt text](/images/result2.png)
![alt text](/images/result3.png)





### Performance 

The performance of the implementation can be inferred from the output logs, i.e. :

    ...
      Testing CCL on image :/home/xib008/xib008/CUCCL/dataset/im229.png
    @ Time elapsed for connectivity 4, GPU : 3.18518  ms
    @ Time elapsed for connectivity 8, GPU : 3.039  ms
    @ Time elapsed for connectivity 4, CPU : 27.4683  ms
    @ Time elapsed for connectivity 8, CPU : 46.0675  ms
    ======================= TEST PASS  =====================

    Testing CCL on image :/home/xib008/xib008/CUCCL/dataset/im22.png
    @ Time elapsed for connectivity 4, GPU : 3.28617  ms
    @ Time elapsed for connectivity 8, GPU : 3.52643  ms
    @ Time elapsed for connectivity 4, CPU : 26.0198  ms
    @ Time elapsed for connectivity 8, CPU : 51.9001  ms
    ======================= TEST PASS  =====================

    Validation Summary : 
    @ Algorithm         : LE
    @ Total Test Images : 1111
    @ Total Pass Images : 1111
    @ Average Time per Image Connection-4, GPU (ms) : 3.89012
    @ Average Time per Image Connection-8, GPU (ms) : 3.71094
    @ Average Time per Image Connection-4, CPU (ms) : 37.135
    @ Average Time per Image Connection-8, CPU (ms) : 60.3834

### Outputs

The output of the program should be a text file. Each row is the list of pixel indexes of a particular blob. The index is the flattened index of the pixels, following row-major format. The 2-D pixel indexes increase from top-left corner towards the bottom-right corners

Examples are shown in _CUCCL/images/*.txt_

## References

1. K. Hawick, A. Leist and D. Playne, Parallel graph component labelling with GPUs and CUDA, Parallel Computing 36 (12) 655-678 (2010)
2. O. Kalentev, A. Rai, S. Kemnitz and R. Schneider, Connected component labeling on a 2D grid using CUDA, J. Parallel Distrib. Comput. 71 (4) 615-620 (2011)
3. V. M. A. Oliveira and R. A. Lotufo, A study on connected components labeling algorithms using GPUs, SIBGRAPI (2010)
4. https://github.com/foota/ccl








