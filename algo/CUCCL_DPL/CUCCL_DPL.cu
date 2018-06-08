#ifndef CUCCL_DPL_CU
#define CUCCL_DPL_CU

#include <host_defines.h>
#include "CUCCL_DPL.cuh"
#include <device_launch_parameters.h>
#include <cmath>
#include <cuda_runtime_api.h>
#include <iostream>
#include <iomanip>


namespace CUCCL{

// catch CUDA error
#define CHECK(call)                                                            \
{                                                                              \
	const cudaError_t error = call;                                            \
	if (error != cudaSuccess)                                                  \
	{                                                                          \
		fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
		fprintf(stderr, "code: %d, reason: %s\n", error,                       \
			cudaGetErrorString(error));                                    \
		exit(1);                                                               \
	}                                                                          \
}

// predefine the block size, can be changed
const int BLOCK = 16;


// initial the label list: labels[i] = i
void init_label_list(int *labelList, int width, int height)
{
	int idx;
	for (int row=0; row<height; row++)
	{
		for (int col=0; col<width; col++)
		{
			idx = col + row*width;
			labelList[idx] = idx;
		}
	}

}

// set the kernel geometry
void set_kernel_dim(int width, int height, dim3 &block, dim3 &grid)
{
    grid.x = width / block.x;
    grid.y = height / block.y;
    if(width % block.x != 0)
      	grid.x++;
    if(height % block.y != 0)
      	grid.y++;
}

// print the initial labels
void print_init_labels(int width, int height, int* labels)
{
	std::cout << "Init labels:" << std::endl;
	for (auto i = 0; i < height; ++i)
	{
		for (auto j = 0; j < width; ++j)
		{
			std::cout << std::setw(3) << labels[i * width + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

// kernel for 4-connectivity
__global__ void dpl_kernel_4(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre)
{
	// thread
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockDim.y*blockIdx.y + ty;
	int col = blockDim.x*blockIdx.x + tx;

	// data
	int W = dataWidth;
	int H = dataHeight;

	// thread index, one row each thread
	int idx = col + row * W;
	if (idx >= W || idx >= H)
		return;
	

	int lowestLabel;
	
	// four loops, one going each direction for each dimension
	// first dimension, vertical, from top to bottom
      if (idx < W)
      {
		lowestLabel = gLabel[idx];
		int i = idx + W;

		while (i<W * H + idx)
		
            {
			if (abs(gData[i]-gData[i - W]) <= thre && lowestLabel < gLabel[i])
                  {
                        gLabel[i] = lowestLabel;
                        *isChanged = true;
                  }
			else 
				lowestLabel = gLabel[i];
			i += W;
            }
      }     
      __syncthreads();

	// first dimension, horizonal, from left to right
      if (idx < H)
      {
		lowestLabel = gLabel[idx*W];
		int i = idx*W + 1;
		
		while (i<(idx+1)*W)
            {
			if (abs(gData[i]-gData[i - 1]) <= thre && lowestLabel < gLabel[i])
                  {
                        gLabel[i] = lowestLabel;
                        *isChanged = true;
                  }
			else 
				lowestLabel = gLabel[i];
			i++;
            }
      }
      __syncthreads();

	// second dimension, vertical, from bottom to top
      if (idx < W) 
      {
		lowestLabel = gLabel[W * (H - 1) + idx];
		int i = W * (H - 2)+idx;

		while (i>=idx)
            {
			if (abs(gData[i]-gData[i + W]) <= thre && lowestLabel < gLabel[i])
                  {
                        gLabel[i] = lowestLabel;
                        *isChanged = true;
                  }
			else 
				lowestLabel = gLabel[i];

			i -= W;
            }
      }
      __syncthreads();
	
	// second dimension, horizonal, from right to left
      if (idx < H)
      {
		lowestLabel = gLabel[(idx + 1) * W - 1];
		int i = (idx + 1) * W - 2;

		while (i>=idx)
            {
                  if (abs(gData[i]-gData[i + 1]) <= thre && lowestLabel < gLabel[i])
                  {
                        gLabel[i] = lowestLabel;
                        *isChanged = true;
                  }
			else 
				lowestLabel = gLabel[i];

			i--;
            }
      }
      __syncthreads();

}


// kernel for 8-connectivity
__global__ void dpl_kernel_8(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre)
{
	// thread
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockDim.y*blockIdx.y + ty;
	int col = blockDim.x*blockIdx.x + tx;

	// data
	int W = dataWidth;
	int H = dataHeight;

	// thread index, one diagonal row each thread
	int idx = col + row * W;
	if (idx >= W + H - 1) 
		return;

	int id, step, lowestLabel;

	// four loops, one going each direction for each dimension
	// first direction, from top/left to right/bottom
	id = (idx < W) ? idx : (idx - W + 1) * W;
	step = W + 1;
	lowestLabel = gLabel[id];
	while (true)
	{
		if (id % W == (W-1) || id / W ==  (H-1))
			break;
		if (abs(gData[id+step]-gData[id]) <= thre && lowestLabel < gLabel[id+step])
		{
			gLabel[id+step] = lowestLabel;
			*isChanged = true;
		}
		else 
			lowestLabel = gLabel[id+step];
		id += step;
	}
	__syncthreads();

	// second direction, from bottom/left to right/top
	id = (idx < W) ? W * (H - 1) + idx : (idx - W + 1) * W;
	step = -W + 1;
	lowestLabel = gLabel[id];
	while (true)
	{
		if ( id % W == (W-1) || id / W == 0)
			break;
		if (abs(gData[id+step]-gData[id]) <= thre && lowestLabel < gLabel[id+step])
		{
			gLabel[id+step] = lowestLabel;
			*isChanged = true;
		}
		else 
			lowestLabel = gLabel[id+step];
		
		id += step;
	}
	__syncthreads();

	// third direction, from bottom/right to left/top
	id = (idx < W) ? W * (H - 1) + idx : (idx - W) * W + W - 1;
	step = -(W + 1);
	lowestLabel = gLabel[id];
	while (true)
	{
		if ( id % W == 0|| id / W == 0)
			break;
		if (abs(gData[id+step]-gData[id]) <= thre && lowestLabel < gLabel[id+step])
		{
			gLabel[id+step] = lowestLabel;
			*isChanged = true;
		}
		else 
			lowestLabel = gLabel[id+step];
		
		id += step;
	}
	__syncthreads();

	// fourth direction, from top/right to left/bottom
	id = (idx < W) ? idx : (idx - W + 1) * W + W - 1;
	step = W - 1;
	lowestLabel = gLabel[id];
	while (true)
	{
		if ( id % W == 0 || id / W == (H-1))
			break;
		if (abs(gData[id+step]-gData[id]) <= thre && lowestLabel < gLabel[id+step])
		{
			gLabel[id+step] = lowestLabel;
			*isChanged = true;
		}
		else 
			lowestLabel = gLabel[id+step];
		id += step;
	}
	__syncthreads();
}



void CCLDPLGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char thre)
{
	// set the device
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("> Starting at Device %d: %s\n\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	
	std::cout << "CUDA DPL..." << std::endl;

	auto nSamples = width * height;

	// allocate data on device
	cudaMalloc((void**)&gData, sizeof(unsigned char) * nSamples);
	CHECK(cudaPeekAtLastError());
	cudaMalloc((void**)&gLabelList, sizeof(int) * nSamples);
	CHECK(cudaPeekAtLastError());

	cudaMemcpy(gData, frame, sizeof(unsigned char) * nSamples, cudaMemcpyHostToDevice);
	CHECK(cudaPeekAtLastError());

	bool* isChanged;
	cudaMalloc((void**)&isChanged, sizeof(bool));
	CHECK(cudaPeekAtLastError());

	// initialize the label list
	init_label_list(labels, width, height);

	// print out the initial labels
	print_init_labels(width, height, labels);

	cudaMemcpy(gLabelList, labels, sizeof(int)*nSamples, cudaMemcpyHostToDevice);
	CHECK(cudaPeekAtLastError());

	// set kernel dimension
	dim3 grid;
	dim3 block(BLOCK, BLOCK, 1);
	set_kernel_dim(width, height, block, grid);
	bool flagHost = true;

	// revoke the kernel multiple times, iterates until there is no change of the labels
	while (flagHost)
	{
		flagHost = false;
		cudaMemcpy(isChanged, &flagHost, sizeof(bool), cudaMemcpyHostToDevice);

		dpl_kernel_4<<< grid, block>>>(gData, gLabelList, width, height, isChanged, thre);
		if (degreeOfConnectivity == 8)
		{
			dpl_kernel_8<<< grid, block>>>(gData, gLabelList, width, height, isChanged, thre);
		}

		cudaMemcpy(&flagHost, isChanged, sizeof(bool), cudaMemcpyDeviceToHost);
		CHECK(cudaPeekAtLastError());

		cudaThreadSynchronize();
	}

	// copy back the labeling results
	cudaMemcpy(labels, gLabelList, sizeof(int) * nSamples, cudaMemcpyDeviceToHost);
	CHECK(cudaPeekAtLastError());

	CHECK(cudaFree(gData));
	CHECK(cudaFree(gLabelList));

	CHECK(cudaDeviceReset());
}
}

#endif