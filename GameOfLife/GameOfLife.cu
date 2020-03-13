/*
	Grant Ludwig
	CPSC 4600, Seattle University
	GameOfLife.cu
	Runs Conways Game of Life
	First runs a serial CPU iteration then parallel using CUDA
	The game board cyclic
	3/15/20

	Initially developed by Grant Ludwig
	Implementation improved based off of NightElfik code: https://github.com/NightElfik/Game-of-life-CUDA
		http://www.marekfiser.com/Projects/Conways-Game-of-Life-on-GPU-using-CUDA
 */

#include <cuda_runtime.h> // CUDA
#include <device_launch_parameters.h> // CUDA
#include <stdio.h> // for IO operations
#include <stdlib.h> // std libary
#include <assert.h>
#include <time.h>
#include <algorithm> // for swap

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

/*
	Runs on the GPU
	Will calculate the next generation

	@param lifeData uChar* array with the current life data
	@param resultLifeData uChar* array with the resulting life data
	@param worldWidth uint
	@param worldHeight uint
*/
__global__ void nextGenCuda(const uchar* lifeData, uchar* resultLifeData, uint worldWidth, uint worldHeight) {
	uint worldSize = worldWidth * worldHeight;

	// will loop through every cell that the thread needs to complete
	// blockIdx.x: block index
	// blockDim.x: threads per block
	// gridDim.x: blocks in grid
	// blockDim.x * gridDim.x: threads per grid
	for (uint cellId = __mul24(blockIdx.x, blockDim.x) + threadIdx.x; cellId < worldSize; cellId += blockDim.x * gridDim.x) { 
		uint x = cellId % worldWidth;
		uint yAbs = cellId - x;
		uint xLeft = (x + worldWidth - 1) % worldWidth;
		uint xRight = (x + 1) % worldWidth;
		uint yAbsUp = (yAbs + worldSize - worldWidth) % worldSize;
		uint yAbsDown = (yAbs + worldWidth) % worldSize;

		// adds up all alive cells surrounding this cell
		uchar aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp]
			+ lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
			+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

		// sets the result cell to the correct value
		resultLifeData[x + yAbs] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
	}
}

/*
	Runs on the CPU
	Will calculate the next generation

	@param lifeData uChar* array with the current life data
	@param resultLifeData uChar* array with the resulting life data
	@param worldWidth uint
	@param worldHeight uint
*/
void nextGenCpu(const uchar* lifeData, uchar* resultLifeData, uint worldWidth, uint worldHeight) {
	for (uint y = 0; y < worldHeight; y++) {
		uint yAbsUp = ((y + worldHeight - 1) % worldHeight) * worldWidth;
		uint yAbs = y * worldWidth;
		uint yAbsDown = ((y + 1) % worldHeight) * worldWidth;

		for (uint x = 0; x < worldWidth; x++) {
			uint xLeft = (x + worldWidth - 1) % worldWidth;
			uint xRight = (x + 1) % worldWidth;

			// adds up all alive cells surrounding this cell
			uchar aliveCells = lifeData[xLeft + yAbsUp] + lifeData[x + yAbsUp] + lifeData[xRight + yAbsUp]
				+ lifeData[xLeft + yAbs] + lifeData[xRight + yAbs]
				+ lifeData[xLeft + yAbsDown] + lifeData[x + yAbsDown] + lifeData[xRight + yAbsDown];

			// sets the result cell to the correct value
			resultLifeData[yAbs + x] = aliveCells == 3 || (aliveCells == 2 && lifeData[x + yAbs]) ? 1 : 0;
		}
	}
}

/*
	Runs the game for CUDA
	Runs as long as the executionTime

	@param d_lifeData uChar*& array with the current life data for the device (GPU)
	@param resultLifeData uChar*& array with the resulting life data for the device (GPU)
	@param worldWidth uint
	@param worldHeight uint
	@param threadsCount ushort
	@param executionTime int How long to run iterations for
	@returns size_t of the number of iterations complete in the executionTime
*/
size_t runGameCuda(uchar*& d_lifeData, uchar*& d_lifeDataBuffer, size_t worldWidth, size_t worldHeight, ushort threadsCount, int executionTime) {
	assert((worldWidth * worldHeight) % threadsCount == 0); // if this is not true, program stops. Need for nice block count
	size_t blockCount = (size_t)((worldWidth * worldHeight) / threadsCount);

	clock_t begin = clock();
	size_t numIterations = 0;
	while (((double)(clock() - begin) / CLOCKS_PER_SEC) < executionTime) {
		nextGenCuda << <blockCount, threadsCount >> > (d_lifeData, d_lifeDataBuffer, worldWidth, worldHeight);
		std::swap(d_lifeData, d_lifeDataBuffer);
		numIterations++;
	}
	return numIterations;
}

/*
	Runs the game for in serial for the CPU
	Runs as long as the executionTime

	@param lifeData uChar*& array with the current life data
	@param resultLifeData uChar*& array with the resulting life data
	@param worldWidth uint
	@param worldHeight uint
	@param threadsCount ushort
	@param executionTime int How long to run iterations for
	@returns size_t of the number of iterations complete in the executionTime
*/
size_t runGameCpu(uchar*& lifeData, uchar*& lifeDataBuffer, size_t worldWidth, size_t worldHeight, int executionTime) {
	clock_t begin = clock();
	size_t numIterations = 0;
	while (((double)(clock() - begin) / CLOCKS_PER_SEC) < executionTime) {
		nextGenCpu(lifeData, lifeDataBuffer, worldWidth, worldHeight);
		std::swap(lifeData, lifeDataBuffer);
		numIterations++;
	}
	return numIterations;
}

/*
	Main Program (Driver)
*/
int main()
{
	// setup data
	// ==========
	const size_t WORLD_WIDTH = 100;
	const size_t WORLD_HEIGHT = 100;
	const int ITERATION_TIME = 10;
	const ushort NUM_THREADS = 1000; //GPU threads

	size_t worldSize = WORLD_WIDTH * WORLD_HEIGHT;
	size_t size_data = sizeof(uchar) * WORLD_WIDTH * WORLD_HEIGHT;

	time_t seedTime = time(0);

	FILE *fp;
	fp = fopen("output.txt", "w+"); // open output file

	// serial using CPU
	// ===================

	printf("Serial: Setting up data for game\n");

	uchar *lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	uchar *lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	srand(seedTime); // set seed for random
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++)
			lifeData[i * WORLD_WIDTH + j] = rand() % 2;
	}

	// write to output
	fprintf(fp, "Serial First Iteration:\n");
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (lifeData[j * WORLD_WIDTH + i] == 1)
				fprintf(fp, "*");
			else
				fprintf(fp, "_");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	printf("Serial: Running game\n");

	size_t iterationsComplete = runGameCpu(lifeData, lifeDataBuffer, WORLD_WIDTH, WORLD_HEIGHT, ITERATION_TIME);

	printf("Serial: Game complete\n");
	printf("Serial completed %zu iterations in %d seconds on a world size of %zu\n\n", iterationsComplete, ITERATION_TIME, worldSize);

	// write to output
	fprintf(fp, "Serial %zu Iteration:\n", iterationsComplete);
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (lifeData[j * WORLD_WIDTH + i] == 1)
				fprintf(fp, "*");
			else
				fprintf(fp, "_");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	free(lifeData);
	free(lifeDataBuffer);

	// parallel using CUDA
	// ===================

	printf("Parallel: Setting up data for game\n");

	// init host arrays
	uchar *h_lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	uchar *h_lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	srand(seedTime); // set seed for random to get same values as serial
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++)
			h_lifeData[i * WORLD_WIDTH + j] = rand() % 2;
	}

	// write to output
	fprintf(fp, "Parallel First Iteration:\n");
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeData[j * WORLD_WIDTH + i] == 1)
				fprintf(fp, "*");
			else
				fprintf(fp, "_");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	//init device arrays
	uchar *d_lifeData, *d_lifeDataBuffer;
	cudaMalloc(reinterpret_cast<void **>(&d_lifeData), size_data);
	cudaMalloc(reinterpret_cast<void **>(&d_lifeDataBuffer), size_data);

	cudaMemcpy(d_lifeData, h_lifeData, size_data, cudaMemcpyHostToDevice);
	cudaMemcpy(d_lifeDataBuffer, h_lifeDataBuffer, size_data, cudaMemcpyHostToDevice);

	printf("Parallel: Running game\n");

	iterationsComplete = runGameCuda(d_lifeData, d_lifeDataBuffer, WORLD_HEIGHT, WORLD_HEIGHT, NUM_THREADS, ITERATION_TIME);

	printf("Parallel: Game complete\n");
	printf("Parallel completed %zu iterations in %d seconds on a world size of %zu\n\n", iterationsComplete, ITERATION_TIME, worldSize);

	cudaMemcpy(h_lifeDataBuffer, d_lifeData, size_data, cudaMemcpyDeviceToHost); // to get final

	// write to output
	fprintf(fp, "Parallel %zu Iteration:\n", iterationsComplete);
	for (size_t i = 0; i < WORLD_HEIGHT; i++) {
		for (size_t j = 0; j < WORLD_WIDTH; j++) {
			if (h_lifeDataBuffer[j * WORLD_WIDTH + i] == 1)
				fprintf(fp, "*");
			else
				fprintf(fp, "_");
		}
		fprintf(fp, "\n");
	}
	fprintf(fp, "\n");

	free(h_lifeData);
	free(h_lifeDataBuffer);

	cudaFree(d_lifeData);
	cudaFree(d_lifeDataBuffer);

	fclose(fp); // close output file

	#pragma region old_code

	//// serial using CPU
	//// ===================

	//printf("Serial: Setting up data for game\n");

	//uchar *lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	//uchar *lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	//memset(lifeData, 0, size_data); // initilize everything to 0

	//lifeData[3 * WORLD_WIDTH + 0] = 1;
	//lifeData[4 * WORLD_WIDTH + 0] = 1;
	//lifeData[5 * WORLD_WIDTH + 0] = 1;

	//lifeData[0 * WORLD_WIDTH + 3] = 1;
	//lifeData[0 * WORLD_WIDTH + 4] = 1;
	//lifeData[0 * WORLD_WIDTH + 5] = 1;

	///*for (size_t i = 0; i < WORLD_WIDTH; i++) {

	//	for (size_t j = 0; j < WORLD_WIDTH; j++) {
	//		if (lifeData[j * WORLD_WIDTH + i] == 1)
	//			printf("*");
	//		else
	//			printf("_");
	//	}
	//	printf("\n");
	//}
	//printf("\n");*/

	//printf("Serial: Running game\n");
	//clock_t begin = clock();

	//runGameCpu(lifeData, lifeDataBuffer, WORLD_WIDTH, WORLD_HEIGHT, NUM_ITERATIONS);

	//clock_t end = clock();
	//double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Serial: Game complete\n");
	//printf("Took %f sec for %zu iterations world size of %zu\n\n", time_spent, NUM_ITERATIONS, worldSize);

	///*for (size_t i = 0; i < WORLD_WIDTH; i++) {

	//	for (size_t j = 0; j < WORLD_WIDTH; j++) {
	//		if (lifeData[j * WORLD_WIDTH + i] == 1)
	//			printf("*");
	//		else
	//			printf("_");
	//	}
	//	printf("\n");
	//}
	//printf("\n");*/

	//free(lifeData);
	//free(lifeDataBuffer);

	//// parallel using CUDA
	//// ===================

	//printf("Parallel: Setting up data for game\n");

	//// init host arrays
	//uchar *h_lifeData = reinterpret_cast<uchar *>(malloc(size_data));
	//uchar *h_lifeDataBuffer = reinterpret_cast<uchar *>(malloc(size_data));

	//memset(h_lifeData, 0, size_data); // initilize everything to 0

	//h_lifeData[3 * WORLD_WIDTH + 0] = 1;
	//h_lifeData[4 * WORLD_WIDTH + 0] = 1;
	//h_lifeData[5 * WORLD_WIDTH + 0] = 1;

	//h_lifeData[0 * WORLD_WIDTH + 3] = 1;
	//h_lifeData[0 * WORLD_WIDTH + 4] = 1;
	//h_lifeData[0 * WORLD_WIDTH + 5] = 1;

	///*for (size_t i = 0; i < WORLD_WIDTH; i++) {

	//	for (size_t j = 0; j < WORLD_WIDTH; j++) {
	//		if (h_lifeData[j * WORLD_WIDTH + i] == 1)
	//			printf("*");
	//		else
	//			printf("_");
	//	}
	//	printf("\n");
	//}
	//printf("\n");*/

	////init device arrays
	//uchar *d_lifeData, *d_lifeDataBuffer;
	//cudaMalloc(reinterpret_cast<void **>(&d_lifeData), size_data);
	//cudaMalloc(reinterpret_cast<void **>(&d_lifeDataBuffer), size_data);

	//cudaMemcpy(d_lifeData, h_lifeData, size_data, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_lifeDataBuffer, h_lifeDataBuffer, size_data, cudaMemcpyHostToDevice);

	//printf("Parallel: Running game\n");
	//begin = clock();

	//runGameCuda(d_lifeData, d_lifeDataBuffer, WORLD_HEIGHT, WORLD_HEIGHT, NUM_ITERATIONS, NUM_THREADS);

	//end = clock();
	//time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
	//printf("Parallel: Game complete\n");
	//printf("Took %f sec for %zu iterations world size of %zu\n\n", time_spent, NUM_ITERATIONS, worldSize);

	//cudaMemcpy(h_lifeDataBuffer, d_lifeData, size_data, cudaMemcpyDeviceToHost); // to get final

	///*for (size_t i = 0; i < WORLD_WIDTH; i++) {

	//	for (size_t j = 0; j < WORLD_WIDTH; j++) {
	//		if (h_lifeDataBuffer[j * WORLD_WIDTH + i] == 1)
	//			printf("*");
	//		else
	//			printf("_");
	//	}
	//	printf("\n");
	//}
	//printf("\n");*/

	//free(h_lifeData);
	//free(h_lifeDataBuffer);

	//cudaFree(d_lifeData);
	//cudaFree(d_lifeDataBuffer);

	#pragma endregion old_code

	return 0;
}