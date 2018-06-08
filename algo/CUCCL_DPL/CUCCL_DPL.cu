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

const int BLOCK = 8;

__global__ void init_CCLDPL(int gLabel[], int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;

	gLabel[id] = id;
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
	

	int label;
      
      if (idx < W)
      {
		label = gLabel[idx];
		int i = idx + W;

		while (i<W * H + idx)
		
            // for (int i = idx + W; i < W * H + idx; i += W)
            {
			if (abs(gData[i]-gData[i - W]) <= thre && label < gLabel[i])
                  {
                        gLabel[i] = label;
                        *isChanged = true;
                  }
			else 
				label = gLabel[i];
			i += W;
            }
      }     
      __syncthreads();


      if (idx < H)
      {
		label = gLabel[idx*W];
		int i = idx*W + 1;
		
		while (i<(idx+1)*W)
            // for (int i = idx*W + 1; i < (idx+1)*W; i ++)
            {
			if (abs(gData[i]-gData[i - 1]) <= thre && label < gLabel[i])
                  {
                        gLabel[i] = label;
                        *isChanged = true;
                  }
			else 
				label = gLabel[i];
			i++;
            }
      }
      __syncthreads();


      if (idx < W) 
      {
		label = gLabel[W * (H - 1) + idx];
		int i = W * (H - 2)+idx;

		while (i>=idx)
            // for (int i = W * (H - 2)+idx; i >= idx; i -= W)
            {
			if (abs(gData[i]-gData[i + W]) <= thre && label < gLabel[i])
                  {
                        gLabel[i] = label;
                        *isChanged = true;
                  }
			else 
				label = gLabel[i];

			i -= W;
            }
      }
      __syncthreads();
	
	
      if (idx < H)
      {
		label = gLabel[(idx + 1) * W - 1];
		int i = (idx + 1) * W - 2;

		while (i>=idx)
            // for (int i = (idx + 1) * W - 2; i >= idx * W; i--)
            {
                  if (abs(gData[i]-gData[i + 1]) <= thre && label < gLabel[i])
                  {
                        gLabel[i] = label;
                        *isChanged = true;
                  }
			else 
				label = gLabel[i];

			i--;
            }
      }
      __syncthreads();

}


__global__ void kernelDPL8(int dirId, unsigned char gData[], int gLabel[], bool* isChanged, int N, int width, int height, int thre)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height)
		return;

	int id = x + y * width;
	int H = N / width;
	int S, E1, E2, step;
	switch (dirId)
	{
	case 0:
		if (id >= width + H - 1) return;
		if (id < width) S = id;
		else S = (id - width + 1) * width;
		E1 = width - 1; // % W
		E2 = H - 1; // / W
		step = width + 1;
		break;
	case 1:
		if (id >= width + H - 1) return;
		if (id < width) S = width * (H - 1) + id;
		else S = (id - width + 1) * width;
		E1 = width - 1; // % W
		E2 = 0; // / W
		step = -width + 1;
		break;
	case 2:
		if (id >= width + H - 1) return;
		if (id < width) S = width * (H - 1) + id;
		else S = (id - width) * width + width - 1;
		E1 = 0; // % W
		E2 = 0; // / W
		step = -(width + 1);
		break;
	case 3:
		if (id >= width + H - 1) return;
		if (id < width) S = id;
		else S = (id - width + 1) * width + width - 1;
		E1 = 0; // % W
		E2 = H - 1; // / W
		step = width - 1;
		break;
	}

	if (E1 == S % width || E2 == S / width)
		return;
	int label = gLabel[S];
	for (int n = S + step;; n += step)
	{
		if (abs(gData[n]-gData[n - step]) <= thre && label < gLabel[n])
		{
			gLabel[n] = label;
			*isChanged = true;
		}
		else label = gLabel[n];
		if (E1 == n % width || E2 == n / width)
			break;
	}
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

		for (int i = 0; i < 4; i++)
		{
			dpl_kernel_4<<< grid, threads>>>(FrameDataOnDevice, LabelListOnDevice, width, height, isChanged, thre);
			if (degreeOfConnectivity == 8)
			{
				kernelDPL8<<< grid, threads>>>(i, FrameDataOnDevice, LabelListOnDevice, isChanged, N, width, height, thre);
			}
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