
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdint.h>
#include <cstdint>
#include <assert.h>
#include <algorithm>
#include <time.h>

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

__global__ void nextGenCuda(const uchar* lifeData, uint worldWidth, uint worldHeight, uchar* resultLifeData) {
	uint worldSize = worldWidth * worldHeight;

	for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) {
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

void nextGenCpu(const uchar* lifeData, uint worldWidth, uint worldHeight, uchar* resultLifeData) {
	for (size_t y = 0; y < worldHeight; ++y) {
		size_t y0 = ((y + worldHeight - 1) % worldHeight) * worldWidth;
		size_t y1 = y * worldWidth;
		size_t y2 = ((y + 1) % worldHeight) * worldWidth;

		for (size_t x = 0; x < worldWidth; ++x) {
			size_t x0 = (x + worldWidth - 1) % worldWidth;
			size_t x2 = (x + 1) % worldWidth;

			uchar aliveCells = lifeData[x0 + y0] + lifeData[x + y0] + lifeData[x2 + y0]
				+ lifeData[x0 + y1] + lifeData[x2 + y1]
				+ lifeData[x0 + y2] + lifeData[x + y2] + lifeData[x2 + y2];
			resultLifeData[y1 + x] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + y1]) ? 1 : 0;
		}
	}
}

void runGameCpu(uchar*& lifeData, uchar*& lifeDataBuffer, size_t worldWidth, size_t worldHeight, size_t iterationsCount) {
	for (size_t i = 0; i < iterationsCount; i++) {
		nextGenCpu(lifeData, worldWidth, worldHeight, lifeDataBuffer);
		std::swap(lifeData, lifeDataBuffer);
	}
}

void runGameCuda(uchar*& d_lifeData, uchar*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight, size_t iterationsCount, ushort threadsCount) {
	assert((worldWidth * worldHeight) % threadsCount == 0);
	size_t reqBlocksCount = (worldWidth * worldHeight) / threadsCount;
	ushort blocksCount = (ushort)std::min((size_t)32768, reqBlocksCount);

	for (size_t i = 0; i < iterationsCount; i++) {
		nextGenCuda << <blocksCount, threadsCount >> > (d_lifeData, worldWidth, worldHeight, d_lifeDataBuffer);
		std::swap(d_lifeData, d_lifeDataBuffer);
	}
}

int main()
{
	// setup data
	// ==========
	const size_t WORLD_WIDTH = 100;
	const size_t WORLD_HEIGHT = 100;
	const size_t NUM_ITERATIONS = 1000000; // 1,000,000
	const ushort NUM_THREADS = 2000; //GPU threads

	size_t worldSize = WORLD_WIDTH * WORLD_HEIGHT;
	size_t size_data = sizeof(uchar) * WORLD_WIDTH * WORLD_HEIGHT;

	// serial using CPU
	// ===================

	printf("Serial: Setting up data for game\n");

	uchar *lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	uchar *lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	memset(lifeData, 0, size_data); // initilize everything to 0

	lifeData[3 * WORLD_WIDTH + 0] = 1;
	lifeData[4 * WORLD_WIDTH + 0] = 1;
	lifeData[5 * WORLD_WIDTH + 0] = 1;

	lifeData[0 * WORLD_WIDTH + 3] = 1;
	lifeData[0 * WORLD_WIDTH + 4] = 1;
	lifeData[0 * WORLD_WIDTH + 5] = 1;

	/*for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (lifeData[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}
	printf("\n");*/

	printf("Serial: Running game\n");
	clock_t begin = clock();

	runGameCpu(lifeData, lifeDataBuffer, WORLD_WIDTH, WORLD_HEIGHT, NUM_ITERATIONS);

	clock_t end = clock();
	double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Serial: Game complete\n");
	printf("Took %f sec for %zu iterations world size of %zu\n\n", time_spent, NUM_ITERATIONS, worldSize);

	/*for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (lifeData[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}
	printf("\n");*/

	free(lifeData);
	free(lifeDataBuffer);

	// parallel using CUDA
	// ===================

	printf("Parallel: Setting up data for game\n");

	// init host arrays
	uchar *h_lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	uchar *h_lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	memset(h_lifeData, 0, size_data); // initilize everything to 0

	h_lifeData[3 * WORLD_WIDTH + 0] = 1;
	h_lifeData[4 * WORLD_WIDTH + 0] = 1;
	h_lifeData[5 * WORLD_WIDTH + 0] = 1;

	h_lifeData[0 * WORLD_WIDTH + 3] = 1;
	h_lifeData[0 * WORLD_WIDTH + 4] = 1;
	h_lifeData[0 * WORLD_WIDTH + 5] = 1;

	/*for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeData[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}
	printf("\n");*/

	//init device arrays
	uchar *d_lifeData, *d_lifeDataBuffer;
	cudaMalloc(reinterpret_cast<void **>(&d_lifeData), size_data);
	cudaMalloc(reinterpret_cast<void **>(&d_lifeDataBuffer), size_data);

	cudaMemcpy(d_lifeData, h_lifeData, size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lifeDataBuffer, h_lifeDataBuffer, size_data, cudaMemcpyHostToDevice);

	printf("Parallel: Running game\n");
	begin = clock();

	runGameCuda(d_lifeData, d_lifeDataBuffer, WORLD_HEIGHT, WORLD_HEIGHT, NUM_ITERATIONS, NUM_THREADS);

	end = clock();
	time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	printf("Parallel: Game complete\n");
	printf("Took %f sec for %zu iterations world size of %zu\n\n", time_spent, NUM_ITERATIONS, worldSize);

	cudaMemcpy(h_lifeDataBuffer, d_lifeData, size_data, cudaMemcpyDeviceToHost); // to get final

	/*for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeDataBuffer[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}
	printf("\n");*/

	free(h_lifeData);
	free(h_lifeDataBuffer);

	cudaFree(d_lifeData);
	cudaFree(d_lifeDataBuffer);

	return 0;
}

/*for (size_t i = 0; i < WORLD_WIDTH; i++) {

		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeData[j * WORLD_WIDTH + i] == 1)
				printf("*");
			else
				printf("_");
		}
		printf("\n");
	}

	printf("\n");*/