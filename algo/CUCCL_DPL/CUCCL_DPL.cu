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

const int BLOCK = 16;

__global__ void init_CCLDPL(int *gLabel, int dataWidth, int dataHeight)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	if (col >= dataWidth || row >= dataHeight)
		return;

	int idx = col + row * dataWidth;

	gLabel[idx] = idx;
}


__global__ void dpl_kernel_4(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockDim.y*blockIdx.y + ty;
	int col = blockDim.x*blockIdx.x + tx;

	int W = dataWidth;
	int H = dataHeight;

	int idx = col + row * W;
	if (idx >= W || idx >= H)
		return;
	

	int lowestLabel;
      
      if (idx < W)
      {
		lowestLabel = gLabel[idx];
		int i = idx + W;

		while (i<W * H + idx)
		
            // for (int i = idx + W; i < W * H + idx; i += W)
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


      if (idx < H)
      {
		lowestLabel = gLabel[idx*W];
		int i = idx*W + 1;
		
		while (i<(idx+1)*W)
            // for (int i = idx*W + 1; i < (idx+1)*W; i ++)
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


      if (idx < W) 
      {
		lowestLabel = gLabel[W * (H - 1) + idx];
		int i = W * (H - 2)+idx;

		while (i>=idx)
            // for (int i = W * (H - 2)+idx; i >= idx; i -= W)
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
	
	
      if (idx < H)
      {
		lowestLabel = gLabel[(idx + 1) * W - 1];
		int i = (idx + 1) * W - 2;

		while (i>=idx)
            // for (int i = (idx + 1) * W - 2; i >= idx * W; i--)
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


__global__ void dpl_kernel_8(unsigned char* gData, int* gLabel, int dataWidth, int dataHeight, bool* isChanged, int thre)
{
	int ty = threadIdx.y;
	int tx = threadIdx.x;
	int row = blockDim.y*blockIdx.y + ty;
	int col = blockDim.x*blockIdx.x + tx;

	int W = dataWidth;
	int H = dataHeight;

	int idx = col + row * W;
	if (idx >= W + H - 1) 
		return;

	int id, step, lowestLabel;

	id = (idx < W) ? idx : (idx - W + 1) * W;
	// E1 = W - 1; // % W
	// E2 = H - 1; // / W
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
	auto N = width * height;

	cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
	cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

	cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

	bool* isChanged;
	cudaMalloc(reinterpret_cast<void**>(&isChanged), sizeof(bool));

	dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
	dim3 threads(BLOCK, BLOCK);

	init_CCLDPL<<<grid, threads >>>(LabelListOnDevice, width, height);

	auto initLabel = static_cast<int*>(malloc(sizeof(int) * width * height));

	cudaMemcpy(initLabel, LabelListOnDevice, sizeof(int) * width * height, cudaMemcpyDeviceToHost);
	std::cout << "Init labels:" << std::endl;
	for (auto i = 0; i < height; ++i)
	{
		for (auto j = 0; j < width; ++j)
		{
			std::cout << std::setw(3) << initLabel[i * width + j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	free(initLabel);

	while (true)
	{
		auto markFalgOnHost = false;
		cudaMemcpy(isChanged, &markFalgOnHost, sizeof(bool), cudaMemcpyHostToDevice);

		dpl_kernel_4<<< grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, width, height, isChanged, thre);
		if (degreeOfConnectivity == 8)
		{
			dpl_kernel_8<<< grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, width, height, isChanged, thre);
		}

		cudaMemcpy(&markFalgOnHost, isChanged, sizeof(bool), cudaMemcpyDeviceToHost);

		if (markFalgOnHost)
		{
			cudaThreadSynchronize();
		}
		else
		{
			break;
		}
	}

	cudaMemcpy(labels, LabelListOnDevice, sizeof(int) * N, cudaMemcpyDeviceToHost);

	cudaFree(FrameDataOnDevice);
	cudaFree(LabelListOnDevice);
}


}

#endif