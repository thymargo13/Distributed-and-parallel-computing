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

#define NUM_BANKS 32
#define LOG_NUM_BANKS 4
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n)\((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

__global__ void blockScan(int *g_odata, int *g_idata, int n){
	extern __shared__ int  temp[];

	int tid = threadIdx.x;
	int offset = 1;

	//load global input into shared memory
	temp[2*tid] = g_idata[2*tid];
	temp[2*tid+1] = g_idata[2*tid+1];

	// build sum in place up the tree
	for(int d = n>>1; d>0; d>>=1){
		__syncthreads();
		if(tid<d){
			int ai = offset *(2*tid+1)-1;
			int bi = offset *(2*tid+2)-1;

			temp[bi] += temp[ai];
		}
		offset *=2;
	}

	//clear the last element
	if(tid==0)
		temp[n-1]=0;

	//traverse down tree & build scan
	for(int d = 1; d<n; d*=2){
		offset >>=1;
		__syncthreads();
		if(tid<d){
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;

			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();

	//write results to device memory
	g_odata[2*tid] = temp[2*tid];
	g_odata[2*tid+1] = temp[2*tid+1];
}

__global__ void blockScanBCAO(int *g_odata, int *g_idata, int n){
	extern __shared__ int  temp[];

	int tid = threadIdx.x;
	int offset = 1;

	//load global input into shared memory
	int ai = tid;
	int bi = tid + (n/2);

	int bankOffsetA = CONFLITCT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLITCT_FREE_OFFSET(bi);

	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];

	// build sum in place up the tree
	for(int d = n>>1; d>0; d>>=1){
		__syncthreads();
		if(tid<d){
			int ai = offset *(2*tid+1)-1;
			int bi = offset *(2*tid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);

			temp[bi] += temp[ai];
		}
		offset *=2;
	}

	//clear the last element
	if(tid==0)
		temp[n-1 + CONFLICT_FREE_OFFSET(n-1)] = 0;

	//traverse down tree & build scan
	for(int d = 1; d<n; d*=2){
		offset >>=1;
		__syncthreads();
		if(tid<d){
			int ai = offset*(2*tid+1)-1;
			int bi = offset*(2*tid+2)-1;
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
__global__ void fullScan(){

}
__global__ void fullScanBCAO(){

}

int main(void){
	return 0;
}

