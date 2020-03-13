
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdint>
#include <assert.h>
#include <algorithm>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

__global__ void lifeCheck(const uchar* lifeData, uint worldWidth, uint worldHeight, uchar* resultLifeData) {
	uint worldSize = worldWidth * worldHeight;

	for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
		cellId < worldSize;
		cellId += blockDim.x * gridDim.x) {
		uint x = cellId % worldWidth;
		uint yAbs = cellId - x;
		uint xLeft = (x + worldWidth - 1) % worldWidth;
		uint xRight = (x + 1) % worldWidth;
		uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
		uint yAbsDown = (yAbs + worldWidth) % worldSize;

		uint aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp] 
			+ lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
			+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

		resultLifeData[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
	}
}

void runLifeCheck(uchar*& d_lifeData, uchar*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
	assert((worldWidth * worldHeight) % threadsCount == 0);
	size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
	ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

	for (size_t i = 0; i < iterationsCount; i++) {
		lifeCheck << <blocksCount, threadsCount >> > (d_lifeData, worldWidth, worldHeight, d_lifeDataBuffer);
		std::swap(d_lifeData, d_lifeDataBuffer);
	}
}

int main()
{
	const size_t WORLD_WIDTH = 10;
	const size_t WORLD_HEIGHT = 10;
	const size_t NUM_ITERATIONS = 100000000;
	const ushort NUM_THREADS = 100;

	size_t size_data = sizeof(uchar) * WORLD_WIDTH * WORLD_HEIGHT;
	// init host arrays
	uchar *h_lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	
	for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			h_lifeData[j * WORLD_WIDTH + i] = 0;
		}
	}

	h_lifeData[3 * WORLD_WIDTH + 3] = 1;
	h_lifeData[4 * WORLD_WIDTH + 3] = 1;
	h_lifeData[5 * WORLD_WIDTH + 3] = 1;

	for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeData[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}

	printf("\n");

	uchar *h_lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	//init device arrays
	uchar *d_lifeData, *d_lifeDataBuffer;
	cudaMalloc(reinterpret_cast<void **>(&d_lifeData), size_data);
	cudaMalloc(reinterpret_cast<void **>(&d_lifeDataBuffer), size_data);

	cudaMemcpy(d_lifeData, h_lifeData, size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lifeDataBuffer, h_lifeDataBuffer, size_data, cudaMemcpyHostToDevice);

	runLifeCheck(d_lifeData, d_lifeDataBuffer, WORLD_HEIGHT, WORLD_HEIGHT, NUM_ITERATIONS, NUM_THREADS);

	cudaMemcpy(h_lifeDataBuffer, d_lifeData, size_data, cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < WORLD_WIDTH; i++) {
		
		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeDataBuffer[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}
	char* name((char*)time_spent);
	printf("Took ");
	printf(name);

	free(h_lifeData);
	free(h_lifeDataBuffer);

	cudaFree(d_lifeData);
	cudaFree(d_lifeDataBuffer);

    return 0;
}
