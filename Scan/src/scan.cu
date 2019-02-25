/*
 * Name: 		Hiu Yin Tang
 * Student ID: 	1831535
 * the Assignment goals achieved: block scan / full scan for large vectors /  Bank conflict avoidance optimization (BCAO)
 * time to execute the different scans on a vector of 10,000,000 entries:
 * Block scan without BCAO: ms
 * Block scan with BCAO: ms
 * Full scan without BCAO: ms
 * Full scan with BCAO: ms
 * The model of CPU:
 * The model of GPU:
 * A short description of any implementation details or performance improvement strategies that you successfully
 * implemented and which improve upon a base level implementation of the target goals.
 *
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

// A helper macro to simplify handling CUDA error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}
#define numElements 1024
#define NUM_BANKS 32
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)\((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__host__ void hostBlockScan(int *h_odata, int* h_idata, int n) {
	h_odata[0] = 0;
	for (int i = 1; i < n; ++n) {
		h_odata[i] = h_idata[i - 1] + h_odata[i - 1];
	}
}

__host__ void hostFullScan(int*h_odata, int*h_idata, int n) {
	int num_blk = 1 + (n - 1) / BLOCK_SIZE;
	for (int i = 0; i < num_blk; i++)
		hostBlockScan(h_odata + i * BLOCK_SIZE, h_idata + i * BLOCK_SIZE,
				min(BLOCK_SIZE, n - i * BLOCK_SIZE));

}

__global__ void blockScan(int *g_odata, int *g_idata, int n) {
	extern __shared__ int temp[];

	int tid = threadIdx.x;
	int offset = 1;

	//load global input into shared memory
	temp[2 * tid] = g_idata[2 * tid];
	temp[2 * tid + 1] = g_idata[2 * tid + 1];

	// build sum in place up the tree
	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	//clear the last element
	if (tid == 0)
		temp[n - 1] = 0;

	//traverse down tree & build scan
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//write results to device memory
	g_odata[2 * tid] = temp[2 * tid];
	g_odata[2 * tid + 1] = temp[2 * tid + 1];
}

__global__ void blockScanBCAO(int *g_odata, int *g_idata, int n) {
	extern __shared__ int temp[];

	int tid = threadIdx.x;
	int offset = 1;

	//load global input into shared memory
	int ai = tid;
	int bi = tid + (n / 2);

	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];

	// build sum in place up the tree
	for (int d = n >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	//clear the last element
	if (tid == 0)
		temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

	//traverse down tree & build scan
	for (int d = 1; d < n; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (tid < d) {
			int ai = offset * (2 * tid + 1) - 1;
			int bi = offset * (2 * tid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//write results to device memory
	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];

}
__global__ void fullScan() {

}
__global__ void fullScanBCAO() {

}

int main(void) {

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Create Host stopwatch timer
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// Create Device timer event objects
	cudaEvent_t start, stop;
	float d_msecs;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Print the vector length to be used, and compute its size

	size_t size = numElements * sizeof(int);
	printf("[BLock Scan of %d elements]\n", numElements);

	//Allocate the host input vector
	int *h_A = (int *) malloc(size);
	//Allocate the host ouput vector
	int *h_O = (int *) malloc(size);
	// Verify that allocations succeeded
	if (h_A == NULL || h_O == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}
	// Initialise the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() % 10;
		//printf("%d\n", h_A[i]);
	}

	// Execute the vector addition on the Host and time it:
	sdkStartTimer(&timer);
	hostblockScan(h_O, h_A, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Executed block scan of %d elements on the Host in = %.5fmSecs\n",
	numElements, h_msecs);

	// Allocate the device input vector A and O
	int *d_A = NULL;
	CUDA_ERROR(cudaMalloc((void ** )&d_A, size),
			"Failed to allocate device vector A");
	int *d_O = NULL;
	CUDA_ERROR(cudaMalloc((void ** )&d_O, size),
			"Failed to allocate device vector O");

	// Copy the host input vectors A in host memory to the device input vectors in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector A from host to device");

	return 0;

}

