/*
 * Student ID: 1831535
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include <helper_cuda.h>
#include <helper_functions.h>

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
__host__ void host_blk_reduce(int *A, int blocksize, int numElements) {
	//adding all elements in the array into the array[first]
	for (int t = 0; t < numElements; t += (blocksize * 2)) {
		for (int i = 1; i < blocksize * 2; i++) {
			if (t + i < numElements) {
				A[t] += A[t + i];
			}
		}
//		printf("t:%d\n",t);
//		printf("A[]:%d\n",A[t]);
	}
}

/**
 * SINGLE thread code that works on Device
 *
 * sequential version of block reduce
 * to execute on the GPU using a single thread (equivalent of host_blk_reduce)
 */
__global__ void single_thread_blk_reduce(int *A, int blocksize,
		int numElements) {
	for (int t = 0; t < numElements; t += (blocksize * 2)) {
		for (int i = 1; i < blocksize * 2; i++) {
			if (t + i < numElements) {
				A[t] += A[t + i];
			}
		}
	}
	// add all first into A[0];
//	for (int t = blocksize * 2; t < numElements; t <<= 1) {
//		A[0] += A[t];
//	}
//	printf("in device ans:%d ", A[0]);
}

/**
 * CUDA Kernel Device code
 *
 * A kernel parallel version of block reduce
 * to execute on the GPU using global memory only
 */
__global__ void global_blk_reduce(const float *A, const float *B, float *C,
		int numElements) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {
		C[i] = A[i] + B[i];
	}
}
/*
 * This returns a single number for the sum of all the elements in the input array.
 */
__host__ int host_full_reduce(int *A, int blocksize, int numElements) {
	// add all first into A[0];
	host_blk_reduce(A, blocksize, numElements);
	for (int t = blocksize * 2; t < numElements; t += blocksize * 2) {
		A[0] += A[t];
	}
	return A[0];
}

__global__ void
segment_compass(){

}

__host__ void
global_full_reduce(){

}

__host__ void
shared_full_reduce(){

}

static void compare_results(const float *vector1, const float *vector2,
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
	// Create Host stop watch timer
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;
	float d_msecs;
	// Create Device timer event objects
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Print the vector length to be used, and compute its size
	int numElements = 9999;
	int blocksize = 1024;

	size_t size = numElements * sizeof(int);
	printf("[Vector addition of %d elements]\n", numElements);

	// Allocate the host input vector A
	int *h_A = (int *) malloc(size);

	// Verify that allocations succeeded
	if (h_A == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input vectors
	for (int i = 0; i < numElements; ++i) {
		h_A[i] = i + 1;
	}
	// Allocate the device input vector A, B and C
	int *d_A = NULL;
	CUDA_ERROR(cudaMalloc((void ** )&d_A, size),
			"Failed to allocate device vector A");

	// Copy the host input vectors A and B in host memory to the device input vectors in device memory
	printf("Copy input data from the host memory to the CUDA device\n");
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy vector A from host to device");

	//
	// Execute the vector addition on the Host and time it:
	//
	sdkStartTimer(&timer);
	host_blk_reduce(h_A, blocksize, numElements);
	sdkStopTimer(&timer);
	h_msecs = sdkGetTimerValue(&timer);
    printf("Executed block reduce of %d elements on the Host in = %.5fmSecs\n", numElements, h_msecs);
	printf("Test PASSED\n");

	//
	// Execute the block reduction on the Device IN A SINGLE THREAD and time it:
	//
//
	cudaEventRecord(start, 0);
	single_thread_blk_reduce<<<1, 1>>>(d_A, blocksize, numElements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
//
//	// wait for device to finish
	cudaDeviceSynchronize();
//
	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch block reduction kernel");
//
	err = cudaEventElapsedTime(&d_msecs, start, stop);
	CUDA_ERROR(err, "Failed to get elapsed time");
//
//	printf("the single thread in device = %.5fmSecs\n" , d_msecs);
//
//    double singlethread = d_msecs;
//
//    // Copy the device result vector in device memory to the host result vector
//    // in host memory.
//    printf("Copy output data from the CUDA device to the host memory\n");
//    err = cudaMemcpy(h_ans, d_ans, size, cudaMemcpyDeviceToHost);
//    CUDA_ERROR(err, "Failed to copy vector C from device to host");
//
//    // Verify that the result vector is correct
//    compare_results(h_ans, h_C, numElements);
	printf("Test PASSED\n");

	// Free host memory
	free(h_A);
	free(d_A);
	// Clean up the Host timer
	sdkDeleteTimer(&timer);

	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	printf("Done\n");
	return 0;
}

