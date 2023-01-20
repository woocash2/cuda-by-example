#include <iostream>
#include <cuda_runtime.h>
#include "cuda_errchk.h"

using namespace std;

const int SIZE = 1024 * 1024 * 100; // 100 MB


// This version is slow because there are a lot of conflicts between threads accessing
// the same histogram entries.
__global__ void computeHistoSlowCUDA(unsigned char * data, int * histo) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int offset = x + y * blockDim.x;

    while (offset < SIZE) {
        atomicAdd(&histo[data[offset]], 1);
        offset += blockDim.x * gridDim.x;
    }
}


// This version is faster because it reduces the traffic by making many partial histograms
// for which only 256 threads concur.
__global__ void computeHistoFastCUDA(unsigned char * data, int * histo) {
    int x = threadIdx.x;
    int y = blockIdx.x;
    int offset = x + y * blockDim.x;

    __shared__ int histoFragment[256];
    histoFragment[x] = 0;

    __syncthreads();

    while (offset < SIZE) {
        atomicAdd(&histoFragment[data[offset]], 1);
        offset += blockDim.x * gridDim.x;
    }

    __syncthreads();

    atomicAdd(&histo[x], histoFragment[x]);
}


int main() {
    // initialising events
    cudaEvent_t start, stop;
    errchk(cudaEventCreate(&start));
    errchk(cudaEventCreate(&stop));
    errchk(cudaEventRecord(start, 0));

    // 100 MB of random bytes
    unsigned char * data = (unsigned char *) malloc(SIZE);

    // host histogram
    int histo[256];

    // device histogram and data
    int * devHisto;
    unsigned char * devData;

    // using memset instead of memcpy
    errchk(cudaMalloc(&devHisto, 256 * sizeof(int)));
    errchk(cudaMalloc(&devData, SIZE));

    // copying host data --> device data
    errchk(cudaMemset(devHisto, 0, 256 * sizeof(int)));
    errchk(cudaMemcpy(devData, data, SIZE, cudaMemcpyHostToDevice));

    // acknowledging how many multiprocessors the GPU has
    cudaDeviceProp props;
    errchk(cudaGetDeviceProperties(&props, 0));

    // and using as many blocks as there are mps * 2
    int blocks = props.multiProcessorCount;

    // kernel call
    computeHistoSlowCUDA<<<2 * blocks, 256>>>(devData, devHisto);
    //computeHistoFastCUDA<<<2 * blocks, 256>>>(devData, devHisto);

    // getting the results back to the host
    errchk(cudaMemcpy(histo, devHisto, 256 * sizeof(int), cudaMemcpyDeviceToHost));    

    // freeing device memory
    cudaFree(devHisto);
    cudaFree(devData);

    // verifying the results
    int sum = 0;
    for (int i = 0; i < 256; i++) {
        sum += histo[i];
    }

    if (sum == SIZE)
        cout << "OK\n";
    else
        cout << "Wrong: " << sum << ' ' << SIZE << '\n';

    // freeing 100MB data
    free(data);

    // measuring time spent on GPU
    errchk(cudaEventRecord(stop, 0));
    errchk(cudaEventSynchronize(stop));
    float elapsedTime;
    errchk(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout << "GPU time: " << elapsedTime << "ms\n";

    // destroying events
    errchk(cudaEventDestroy(start));
    errchk(cudaEventDestroy(stop));

    return 0;
}