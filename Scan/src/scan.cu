/*
 * Name: 		Hiu Yin Tang
 * Student ID: 	1831535
 * the Assignment goals achieved: block scan / full scan for large vectors /  Bank conflict avoidance optimization (BCAO)
 * time to execute the different scans on a vector of 10,000,000 entries:
 * Block scan without BCAO: ms
 * Block scan with BCAO: ms
 * Full scan without BCAO: ms
 * Full scan with BCAO: ms
 * The model of CPU: Intel® Core™ i5-6500 CPU @ 3.20GHz × 4
 * The model of GPU: GeForce GTX 960
 *
 * 1. BLOCK_SIZE refers to 2*BLOCK_SIZE element per block
 * 2. BLOCK_SIZE cannot be higher than 128.
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
#define BLOCK_SIZE 64
#define numElements 10000000
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)\((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

// Host exclusive full scan
__host__ void h_fullScan(int *h_odata, int* h_idata, int n) {
	h_odata[0] = 0;
	for (int i = 1; i < n; ++i) {
		h_odata[i] = h_idata[i - 1] + h_odata[i - 1];
	}
}
// Host exclusive block scan
__host__ void h_blockScan(int*h_odata, int*h_idata, int n) {
	int blk_size = BLOCK_SIZE * 2;
	int num_blk = 1 + n / blk_size; //calculating how many block. e.g. 5000/1024 =4.88 +1 = 5 block in total

	for (int blk = 0; blk < num_blk; blk++) {
		if (blk * blk_size > n)
			blk_size = min(blk_size, n - blk_size);
		h_fullScan(h_odata + blk * blk_size, h_idata + blk * blk_size,
				blk_size);
	}
//	for(int k = 0; k<n; k++)
//		  printf("%d : %d\n",k,h_odata[k]);
}
// Compare the results of host and device
static void compare_results(const int* vector1, const int*vector2, int n) {
	for (int i = 0; i < numElements; ++i) {
		if (fabs(vector1[i] - vector2[i]) > 1e-5f) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}
// Device block scan without BCAO
__global__ void d_blockScan(int *g_odata, int *g_idata, int n) {
	// copy to shared memory
	__shared__ int temp[BLOCK_SIZE * 2];
	const int locallen = BLOCK_SIZE * 2;
	int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	if (i < n)
		temp[threadIdx.x] = g_idata[i];
	i += BLOCK_SIZE;
	if (i < n)
		temp[threadIdx.x + BLOCK_SIZE] = g_idata[i];

	// thread id and offset
	const int thid = threadIdx.x;
	int offset = 1;

	// build sum in place up the tree
	for (int d = locallen >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Clear the last element
	if (thid == 0) {
		temp[locallen - 1] = 0;
	}

	// traverse down tree and build scan
	for (int d = 1; d < locallen; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// copy to output
	if (i < n)
		g_odata[i] = temp[threadIdx.x + BLOCK_SIZE];
	i -= BLOCK_SIZE;
	if (i < n)
		g_odata[i] = temp[threadIdx.x];
}
// Device block scan with BCAO
__global__ void d_blockScanBCAO(int *g_odata, int *g_idata, int n) {
	// copy to shared memory
	__shared__ int temp[BLOCK_SIZE * 2 + CONFLICT_FREE_OFFSET(BLOCK_SIZE * 2)];
	const int locallen = BLOCK_SIZE * 2;
	int i = blockIdx.x * 2 * blockDim.x + threadIdx.x;
	if (i < n)
		temp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)] = g_idata[i];
	i += BLOCK_SIZE;
	if (i < n)
		temp[(threadIdx.x + BLOCK_SIZE)
				+ CONFLICT_FREE_OFFSET((threadIdx.x + BLOCK_SIZE))] = g_idata[i];

	// thread id and offset
	const int thid = threadIdx.x;
	int offset = 1;

	// build sum in place up the tree
	for (int d = locallen >> 1; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}

	// Clear the last element
	if (thid == 0) {
		temp[locallen - 1 + CONFLICT_FREE_OFFSET(locallen - 1)] = 0;
	}

	// traverse down tree and build scan
	for (int d = 1; d < locallen; d *= 2) {
		offset >>= 1;
		__syncthreads();
		if (thid < d) {
			int ai = offset * (2 * thid + 1) - 1;
			int bi = offset * (2 * thid + 2) - 1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	// copy to output
	if (i < n)
		g_odata[i] = temp[(threadIdx.x + BLOCK_SIZE)
				+ CONFLICT_FREE_OFFSET((threadIdx.x + BLOCK_SIZE))];
	i -= BLOCK_SIZE;
	if (i < n)
		g_odata[i] = temp[threadIdx.x + CONFLICT_FREE_OFFSET(threadIdx.x)];
}
// Device Full Scan without BCAO
__global__ void fullScan() {

}
// Device Full Scan with BCAO
__global__ void fullScanBCAO() {

}
int main(void) {

	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// Create Host stop watch timer
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
	//Allocate the host f_output vector
	int *h_fO = (int *) malloc(size);
	//Allocate the host b_output vector
	int *h_bO = (int *) malloc(size);
	//Allocate the host vector for copy the result of device
	int *h_C = (int *) malloc(size);
	// Verify that allocations succeeded
	if (h_A == NULL || h_fO == NULL || h_bO == NULL || h_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vectors
	for (int i = 0; i < numElements; ++i) {
		//h_A[i] = i + 1;
		h_A[i] = rand() % 10;
		//printf("%d\n", h_A[i]);
	}

	//Execute the Full Scan on the Host and time it:
	sdkStartTimer(&timer);
	//Call the host full Scan
	h_fullScan(h_fO, h_A, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Executed Full scan of %d elements on the Host in = %.5fmSecs\n",
	numElements, h_msecs);

	//Execute the BLOCK Scan on the Host and time it:
	sdkStartTimer(&timer);
	//Call the host block Scan
	h_blockScan(h_bO, h_A, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf("Executed block scan of %d elements on the Host in = %.5fmSecs\n",
	numElements, h_msecs);

	/*------------ PARALLEL -----------*/
	// Allocate the device input vector d_A and d_O
	int *d_A = NULL;
	CUDA_ERROR(cudaMalloc((void ** )&d_A, size),
			"Failed to allocate device vector d_A");
	int *d_O = NULL;
	CUDA_ERROR(cudaMalloc((void ** )&d_O, size),
			"Failed to allocate device vector d_O");

	// Copy the host input vectors h_A in host memory to the device input vectors d_A in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector d_A from host to device");

	/*----- Execute the Block Scan on the Device and time it: ----*/
	cudaEventRecord(start, 0);
	//<< blocks per grid, threads per block, size of share memory>>
	int blockspergrid = 1 + (numElements - 1) / BLOCK_SIZE;
	//printf("%d",blockspergrid);
	d_blockScan<<<blockspergrid, BLOCK_SIZE>>>(d_O, d_A, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// wait for device to finish
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");
	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	// Copy the device result vector d_O in device memory to the host result vector in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_O, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_O from device to host");
	printf("Executed Block Scan of %d elements on the Device in = %.5fmSecs\n",
	numElements, d_msecs);
	// Verify that the result vector is correct
	compare_results(h_bO, h_C, numElements);
	printf("Test PASSED\n");


	/*----- Execute the Block Scan with BCAO on the Device and time it: ----*/
	cudaEventRecord(start, 0);
	//<< blocks per grid, threads per block, size of share memory>>
	d_blockScanBCAO<<<blockspergrid, BLOCK_SIZE>>>(d_O, d_A, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	// wait for device to finish
	cudaDeviceSynchronize();
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block scan kernel");
	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
	// Copy the device result vector d_O in device memory to the host result vector in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_O, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector d_O from device to host");
	printf("Executed Block Scan with BCAO of %d elements on the Device in = %.5fmSecs\n",
	numElements, d_msecs);
	// Verify that the result vector is correct
	compare_results(h_bO, h_C, numElements);
	printf("Test PASSED\n");

	/*----- Execute the Full Scan on the Device and time it: ----*/

	/*----- Execute the Full Scan with BCAO on the Device and time it: ----*/

	/*------------ FREE -----------*/
	//Free device global memory
	err = cudaFree(d_A);
	CUDA_ERROR(err, "Failed to free device vector d_A");
	err = cudaFree(d_O);
	CUDA_ERROR(err, "Failed to free device vector d_O");

	//Free host memory
	free(h_A);
	free(h_fO);
	free(h_bO);
	free(h_C);

	//Clear up the Host timer
	sdkDeleteTimer(&timer);

	//Clear up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//Rest the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	return 0;

}
