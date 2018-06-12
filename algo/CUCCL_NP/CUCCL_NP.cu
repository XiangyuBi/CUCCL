#ifndef CUCCL_NP_CU
#define CUCCL_NP_CU


#include "CUCCL_NP.cuh"

#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

namespace CUCCL {

	const int BLOCK = 8;

	__device__ int atom_MIN(int a, int b)
	{
		//atomic operation minimum
		if (a < b)
			return a;
		else
			return b;
	}


	__global__ void InitCCL(int L_d[], int width, int height)
	{
		// interger Ld[N] N = width * height

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int id = x + y * width;

		L_d[id] = id;
		// do in parallel on the device using N threads: initialise Ld[0 . . . N − 1] such that Ld[i] <-- i
	}

	__global__ void kernel4(unsigned char D_d[], int L_d[], bool* m_d, int N, int width, int height, int threshold)
	{
		// This is GPU kernel for 4-conectivity
		// m_d : examine this boolean value to determine if another iteration of the algorithm is required

		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int id = x + y * width; // declare id: id <-- threadID & blockID from CUDA runtime

		int label = D_d[id]; // label <-- L[id]
		int minlabel = N; // N = width * height (max possible label)

						  // Finding minimum connected label 
		bool up = id - width >= 0 && abs(label - D_d[id - width]) <= threshold;
		bool down = id + width < N && abs(label - D_d[id + width]) <= threshold;
		bool left = id % width && abs(label - D_d[id - 1]) <= threshold;
		bool right = id % width + 1 != width && abs(label - D_d[id + 1]) <= threshold;

		// up  
		if (up == true) //n_id[0]
			minlabel = atom_MIN(minlabel, L_d[id - width]);
		// down
		if (down == true)  //n_id[1]
			minlabel = atom_MIN(minlabel, L_d[id + width]);
		// left 
		if (left) //n_id[2]
			minlabel = atom_MIN(minlabel, L_d[id - 1]);
		// right
		if (right) //n_id[3]
			minlabel = atom_MIN(minlabel, L_d[id + 1]);

		if (minlabel < L_d[id]) // Changes happens and another iteration of the algorithm is required
		{
			L_d[id] = minlabel; // L[id] <-- minlabel
			*m_d = true;
		}
	}

	__global__ void kernel8(unsigned char D_d[], int L_d[], bool* m_d, int N, int width, int height, int threshold)
	{
		// This is GPU kernel for 8-conectivity
		// m_d : examine this boolean value to determine if another iteration of the algorithm is required
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;

		if (x >= width || y >= height)
			return;

		int id = x + y * width; // declare id: id <-- threadID & blockID from CUDA runtime

		int label = D_d[id];  // label <-- L[id]
		int minlabel = N; // N = width * height (max possible label)

						  // Finding minimum connected label 
		bool up = id - width >= 0 && abs(label - D_d[id - width]) <= threshold;
		bool down = id + width < N && abs(label - D_d[id + width]) <= threshold;
		bool left = id % width;
		bool right = id % width + 1 != width;
		// up 
		if (up) //n_id[0]
			minlabel = atom_MIN(minlabel, L_d[id - width]);
		// down 
		if (down) //n_id[1]
			minlabel = atom_MIN(minlabel, L_d[id + width]);
		// left 1,2,3
		if (left)
		{
			if (abs(label - D_d[id - 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id - 1]); //n_id[2]
			if (id - width - 1 >= 0 && abs(label - D_d[id - width - 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id - width - 1]);//n_id[3]
			if (id + width - 1 < N && abs(label - D_d[id + width - 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id + width - 1]);//n_id[4]
		}
		// right 1,2,3
		if (right)
		{
			if (abs(label - D_d[id + 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id + 1]);//n_id[5]
			if (id - width + 1 >= 0 && abs(label - D_d[id - width + 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id - width + 1]);//n_id[6]
			if (id + width + 1 < N && abs(label - D_d[id + width + 1]) <= threshold)
				minlabel = atom_MIN(minlabel, L_d[id + width + 1]);//n_id[7]
		}

		if (minlabel < L_d[id]) // Changes happens and another iteration of the algorithm is required
		{
			L_d[id] = minlabel; // L[id] <-- minlabel
			*m_d = true;
		}
	}

	void CCLNPGPU::CudaCCL(unsigned char* frame, int* labels, int width, int height, int degreeOfConnectivity, unsigned char threshold)
	{
		auto N = width * height;

		//declare integer Ld[N]
		cudaMalloc(reinterpret_cast<void**>(&LabelListOnDevice), sizeof(int) * N);
		cudaMalloc(reinterpret_cast<void**>(&FrameDataOnDevice), sizeof(unsigned char) * N);

		cudaMemcpy(FrameDataOnDevice, frame, sizeof(unsigned char) * N, cudaMemcpyHostToDevice);

		bool* m_d; // declare boolean md in device memory
		cudaMalloc(reinterpret_cast<void**>(&m_d), sizeof(bool));

		dim3 grid((width + BLOCK - 1) / BLOCK, (height + BLOCK - 1) / BLOCK);
		dim3 threads(BLOCK, BLOCK);

		InitCCL << <grid, threads >> >(LabelListOnDevice, width, height);
		//do in parallel on the device using N threads: initialise Ld[0 . . . N − 1] such that Ld[i] <-- i

		auto initLabel = static_cast<int*>(malloc(sizeof(int) * width * height));
		// get initial label to print


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
			// repeat
			//		do in parallel on the device using N threads: call Mesh Kernel A(Dd, Ld, md) 
			// until md = false

			auto markFalgOnHost = false;
			cudaMemcpy(m_d, &markFalgOnHost, sizeof(bool), cudaMemcpyHostToDevice);

			if (degreeOfConnectivity == 4)
			{
				kernel4 << < grid, threads >> >(FrameDataOnDevice, LabelListOnDevice, m_d, N, width, height, threshold);
				// Mesh_Kernel_A
				cudaThreadSynchronize();
			}
			else
				kernel8 << < grid, threads >> >(FrameDataOnDevice, LabelListOnDevice, m_d, N, width, height, threshold);
			//Mesh_Kernel_A
			cudaThreadSynchronize();
			cudaMemcpy(&markFalgOnHost, m_d, sizeof(bool), cudaMemcpyDeviceToHost);

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