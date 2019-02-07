/*
 * Student ID: 1831535
 * Name : HY Tang
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// define the block size
#define BLOCK_SIZE 1024

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

/**
 *Host version of Block Reduce
 *the block size is a parameter and the algorithm calculates the reduction of each segment = 2 × block size of the array for each block.
 *The results of this function must be compared against the results of all GPU kernel implementations of block reduce
 *to check validity.
 *
 */
__host__ void host_blk_reduce(const int *A, int *O, int numElements) {
	// copy the array to output
	for (int j = 0; j < numElements; j++) {
		O[j] = A[j];
	}
	//adding all elements in the array into the array[first]
	for (int t = 0; t < numElements; t += (BLOCK_SIZE * 2)) {
		for (int i = 1; i < BLOCK_SIZE * 2; i++) {
			if (t + i < numElements)
				O[t] += O[t + i];
		}
	}

}

/**
 * SINGLE thread code that works on Device
 *
 * sequential version of block reduce
 * to execute on the GPU using a single thread (equivalent of host_blk_reduce)
 */
__global__ void single_thread_blk_reduce(const int *A, int *O,
		int numElements) {
	// copy the array to output
	for (int j = 0; j < numElements; j++) {
		O[j] = A[j];
	}
	//adding all elements in the array into the array[first]
	for (int t = 0; t < numElements; t += (BLOCK_SIZE * 2)) {
		for (int i = 1; i < BLOCK_SIZE * 2; i++) {
			if (t + i < numElements)
				O[t] += O[t + i];
		}
	}

}

/**
 * CUDA Kernel Device code
 *
 * A kernel parallel version of block reduce
 * to execute on the GPU using global memory only
 */
__global__ void global_blk_reduce(int *A, int *O, int numElements) {
	int B[BLOCK_SIZE * 2];

	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		B[threadIdx.x] = A[i];

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (i < stride)
			B[i] += B[i + stride];
	}
	// copy to output
	if (i < numElements)
		O[i] = B[threadIdx.x];
}

__global__ void shared_blk_reduce(int* A, int *O, int numElements) {
	// copying to share memory
	__shared__ int B[BLOCK_SIZE * 2];
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i < numElements)
		B[threadIdx.x] = A[i];

	for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
		__syncthreads();
		if (i < stride)
			B[i] += B[i + stride];
	}
	// copying to the output
	if (i < numElements)
		O[i] = B[threadIdx.x];
}
/*
 * This returns a single number for the sum of all the elements in the input array.
 */
__host__ int host_full_reduce(int *A, int *O, int numElements) {
	// add all first into A[0];
	host_blk_reduce(A,O, numElements);
	for (int t = BLOCK_SIZE * 2; t < numElements; t += BLOCK_SIZE * 2) {
		A[0] += A[t];
	}
	return A[0];
}

/*
 *A kernel to copy the first int of every segment of an array down to the start of the array
 */
__global__ void segment_compass(int *A, int numElements) {

}

/*
 * A HOST CPU function that executes a full reduce of an array using global memory
 * only by making the appropriate sequence of calls to global_blk_reduce and segment_compress
 */
__host__ int global_full_reduce(int* A, int numElements) {
	return 0;
}

/*
 * A HOST CPU function that executes a full reduce of an array using shared memory by
 * making the appropriate sequence of calls to shared_blk_reduce and segment_compress.
 */
__host__ int shared_full_reduce(int* A, int numElements) {
	return 0;
}

static void compare_results(const int *vector1, const int *vector2,
		int numElements) {
	for (int i = 0; i < numElements; ++i) {
		if (fabs(vector1[i] - vector2[i]) > 1e-5f) {
			fprintf(stderr, "Result verification failed at element %d!\n", i);
			exit(EXIT_FAILURE);
		}
	}
}
/**
 * Host main routine
 */
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
	int numElements = 5000;
	size_t size = numElements * sizeof(int);
	printf("[BLock reduction of %d elements]\n", numElements);

	// Allocate the host input vector A
	int *h_A = (int *) malloc(size);
	int *h_C = (int *) malloc(size);
	// Allocate the host output vector that will contain the sum calculate by the Host
	int *h_OUT = (int *) malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL || h_OUT == NULL || h_C == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = rand() / (int) RAND_MAX;
	}

	/*
	 * Execute the block reduce on the Host and time it:
	 */
	sdkStartTimer(&timer);
	host_blk_reduce(h_A, h_OUT, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
	printf(
			"Executed Block Reduction of %d elements on the Host in = %.5fmSecs\n",
			numElements, h_msecs);

	// Allocate the device input vector A and OUTPUT
	int *d_A = NULL;
	err = cudaMalloc((void **) &d_A, size);
	CUDA_ERROR(err, "Failed to allocate device vector A");

	int *d_OUT = NULL;
	err = cudaMalloc((void **) &d_OUT, size);
	CUDA_ERROR(err, "Failed to allocate device vector output");

	// Copy the host input vectors A in host memory to the device input vectors in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector A from host to device");

	/*
	 * Execute the vector addition on the Device IN A SINGLE THREAD and time it:
	 */
	cudaEventRecord(start, 0);
	single_thread_blk_reduce<<<1, 1>>>(d_A, d_OUT, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block reduce kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf(
			"Executed  Block Reduction of %d elements on the Device in a SINGLE THREAD in = %.5fmSecs\n",
			numElements, d_msecs);

	double singlethread = d_msecs;

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector C from device to host");

	// Verify that the result vector is correct
	compare_results(h_OUT, h_C, numElements);
	printf("Test PASSED\n");

	// Note this pattern, based on integer division, for rounding up
	int blocksPerGrid = 1 + ((numElements - 1) / BLOCK_SIZE);

	/*
	 *Execute the Block reduction on the Device in shared memory and time it:
	 */
	cudaEventRecord(start, 0);
	shared_blk_reduce<<<blocksPerGrid, BLOCK_SIZE>>>(d_A, d_OUT, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch shared block reduce kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf(
			"Executed block reduce in SHARE memory of %d elements on the Device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, blocksPerGrid, BLOCK_SIZE, d_msecs);

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector C from device to host");

	// Verify that the result vector is correct
	compare_results(h_OUT, h_C, numElements);
	printf("Test PASSED\n");

	/*
	 *Execute the Block reduction on the Device in shared memory and time it:
	 */
	cudaEventRecord(start, 0);
	global_blk_reduce<<<blocksPerGrid, BLOCK_SIZE>>>(d_A, d_OUT, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	// wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch shared block reduce kernel");

	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	printf(
			"Executed block reduce in GLOBAL memory of %d elements on the Device in %d blocks of %d threads in = %.5fmSecs\n",
			numElements, blocksPerGrid, BLOCK_SIZE, d_msecs);

	// Copy the device result vector in device memory to the host result vector
	// in host memory.
	printf("Copy output data from the CUDA device to the host memory\n");
	err = cudaMemcpy(h_C, d_OUT, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy vector C from device to host");

	// Verify that the result vector is correct
	compare_results(h_OUT, h_C, numElements);
	printf("Test PASSED\n");

}

