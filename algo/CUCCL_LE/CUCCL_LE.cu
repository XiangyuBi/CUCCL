/*
CCL implements of LE kernel
Name: Wenyu Zhang
Email: wez078@ucsd.edu
Date: 6/9/2018
*/

#ifndef CUCCL_LE_CU
#define CUCCL_LE_CU

#include <cmath>
#include <device_launch_parameters.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>

#include "CUCCL_LE.cuh"
using namespace std;
namespace CUCCL{

const int BLOCK = 8;

// find the minor
__device__ int getMinor(int a, int b)
{
	return a < b ? a : b;
}

// distance of two nums
__device__ unsigned char getDiff(unsigned char a, unsigned char b)
{
	return abs(a - b);
}

// initialize equivalence chain, mapping each cell to a thread
__global__ void InitCCL(int labelList[], int reference[], int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height)  // out of range
		return;

	int id = idy * width + idx;

	labelList[id] = reference[id] = id;  // the label of each cell is preset as itself
}

// Phase I. scanning
__global__ void scanning(unsigned char frame[], 
	                     int labelList[], 
	                     int reference[], 
	                     bool* markFlag, 
	                     int N, int width, int height, 
	                     unsigned char threshold)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height)  // out of range
		return;

	int id = idy * width + idx;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && getDiff(value, frame[id - width]) <= threshold)  // not the first row
		label = getMinor(label, labelList[id - width]);
	if (id + width < N  && getDiff(value, frame[id + width]) <= threshold)  // not the last row
		label = getMinor(label, labelList[id + width]);

	int col = id % width;

	if (col > 0 && getDiff(value, frame[id - 1]) <= threshold)              // not the first col
		label = getMinor(label, labelList[id - 1]);
	if (col + 1 < width  && getDiff(value, frame[id + 1]) <= threshold)     // not the last col
		label = getMinor(label, labelList[id + 1]);

	if (label < labelList[id])                                              // update reference
	{
		reference[labelList[id]] = label;
		*markFlag = true;
	}
}

__global__ void scanning8(unsigned char frame[], 
	                      int labelList[], 
	                      int reference[], 
	                      bool* markFlag, 
	                      int N, int width, int height, 
	                      unsigned char threshold)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	int id = idx + idy * blockDim.x * gridDim.x;

	if (id >= N)
		return;

	unsigned char value = frame[id];
	int label = N;

	if (id - width >= 0 && getDiff(value, frame[id - width]) <= threshold)             // not the first row
		label = getMinor(label, labelList[id - width]);                                // compare upper adjacent

	if (id + width < N  && getDiff(value, frame[id + width]) <= threshold)             // not the last row
		label = getMinor(label, labelList[id + width]);                                // compare lower adjacent

	int col = id % width;
	if (col > 0)                                                                       // not the first col
	{
		if (getDiff(value, frame[id - 1]) <= threshold)
			label = getMinor(label, labelList[id - 1]);                                // compare left adjacent
		if (id - width - 1 >= 0 && getDiff(value, frame[id - width - 1]) <= threshold) // boundary can merge..move left one cell
			label = getMinor(label, labelList[id - width - 1]);
		if (id + width - 1 < N  && getDiff(value, frame[id + width - 1]) <= threshold)
			label = getMinor(label, labelList[id + width - 1]);
	}
	if (col + 1 < width)                                                               // not the last col
	{
		if (getDiff(value, frame[id + 1]) <= threshold)                                // compare right adjacent
			label = getMinor(label, labelList[id + 1]);
		if (id - width + 1 >= 0 && getDiff(value, frame[id - width + 1]) <= threshold) // boundary can merge..move right one cell
			label = getMinor(label, labelList[id - width + 1]);
		if (id + width + 1 < N  && getDiff(value, frame[id + width + 1]) <= threshold)
			label = getMinor(label, labelList[id + width + 1]);
	}

	if (label < labelList[id])                                                         // update reference
	{
		reference[labelList[id]] = label;
		*markFlag = true;
	}
}

// Phase II. analysis
__global__ void analysis(int labelList[], int reference[], int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height)
		return;

	int id = idx + idy * width;

	int label = labelList[id];
	int ref;
	if (label == id)
	{
		do
		{
			ref = label;
			label = reference[ref];
		} while (ref ^ label);                    // XOR until equal
		reference[id] = label;
	}
}

// Phase III. labelling
__global__ void labelling(int labelList[], int reference[], int width, int height)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	if (idx >= width || idy >= height)
		return;

	int id = idx + idy * width;

	labelList[id] = reference[reference[labelList[id]]];
}


//  Core function
void CCLLEGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
{
	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&ReferenceOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* markFlagOnDevice;
	cudaMalloc(reinterpret_cast<void**>(&markFlagOnDevice), sizeof(bool));

	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);
    
    // parallel initialization
	InitCCL <<<grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);

	auto initLabel = reinterpret_cast<int*>(malloc(sizeof(int) * width * height));

    // print on Host
	cudaMemcpy(initLabel, LabelListOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	cout << "Init labels:" << endl;
	for (auto i = 0; i < height; ++i)
	{
		for (auto j = 0; j < width; ++j)
		{
			cout << setw(3) << initLabel[i * width + j] << " ";
		}
		cout << endl;
	}
	cout << endl;
	free(initLabel);

	while (true)   // one iteration
	{
		auto markFlagOnHost = false;
		cudaMemcpy(markFlagOnDevice, &markFlagOnHost, sizeof(bool), cudaMemcpyHostToDevice);

		if (degreeOfConnectivity == 4)  
		{  // not at component boundary
			scanning <<< grid, threads >>>(FrameDataOnDevice, 
				                           LabelListOnDevice, 
				                           ReferenceOnDevice, 
				                           markFlagOnDevice, 
				                           N, width, height, 
				                           threshold);
			cudaThreadSynchronize();
		}
		else
			scanning8 <<< grid, threads >>>(FrameDataOnDevice, 
				                            LabelListOnDevice, 
				                            ReferenceOnDevice, 
				                            markFlagOnDevice, 
				                            N, width, height, 
				                            threshold);

		cudaThreadSynchronize();
		cudaMemcpy(&markFlagOnHost, markFlagOnDevice, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFlagOnHost)  
		{  // update
			analysis <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
			cudaThreadSynchronize();
			labelling <<< grid, threads >>>(LabelListOnDevice, ReferenceOnDevice, width, height);
		}
		else
		{
			break;
		}
	}

	cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
	cudaFree(ReferenceOnDevice);
}




}

#endif