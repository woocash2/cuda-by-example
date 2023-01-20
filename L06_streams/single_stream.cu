#include <iostream>
#include <cuda_runtime.h>
#include "cuda_errchk.h"

using namespace std;

const int N = 1024 * 1024;
const int SIZE = N * 20;


__global__ void avg3BlockCUDA(int * a, int * b, int * c) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float as = (a[idx] + a[(idx + 1) % 256] + a[(idx + 2) % 256]) / 3.0f;
        float bs = (b[idx] + b[(idx + 1) % 256] + b[(idx + 2) % 256]) / 3.0f;
        c[idx] = (as + bs) / 2.0f;
    }
}


int main() {
     // init events
    cudaEvent_t start, stop;
    errchk(cudaEventCreate(&start));
    errchk(cudaEventCreate(&stop));
    errchk(cudaEventRecord(start, 0));

    // Get our device properties
    cudaDeviceProp prop;
    int devIdx;
    errchk(cudaGetDevice(&devIdx));
    errchk(cudaGetDeviceProperties(&prop, devIdx));

    // Check if out device handles overlaps
    // (Ability to perform kernels and memory copying host <-> device simultaneously)
    if (!prop.deviceOverlap) {
        cout << "No speedup from streams\n";
        return 0;
    }

    // creating a stream
    cudaStream_t stream;
    errchk(cudaStreamCreate(&stream));

    // host and device buffers
    int * a, * b, * c;
    int * devA, * devB, * devC;

    // allocating dev buffers
    errchk(cudaMalloc(&devA, N * sizeof(int)));
    errchk(cudaMalloc(&devB, N * sizeof(int)));
    errchk(cudaMalloc(&devC, N * sizeof(int)));

    // allocating host buffers. We need cudaHostAlloc because
    // cudaStream requires page-locked memory
    errchk(cudaHostAlloc(&a, SIZE * sizeof(int), cudaHostAllocDefault));
    errchk(cudaHostAlloc(&b, SIZE * sizeof(int), cudaHostAllocDefault));
    errchk(cudaHostAlloc(&c, SIZE * sizeof(int), cudaHostAllocDefault));

    // filling host buffers with random values
    for (int i = 0; i < SIZE; i++) {
        a[i] = rand() % 1000000;
        b[i] = rand() % 1000000;
    }

    // call kernels by chunks
    for (int i = 0; i < SIZE; i += N) {
        // Copy chunks of size N. We use cudaMemcpyAsync as it will turn out being important
        // when we will use more than one stream
        errchk(cudaMemcpyAsync(devA, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
        errchk(cudaMemcpyAsync(devB, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

        // kernel call
        avg3BlockCUDA<<<N / 256, 256, 0, stream>>>(devA, devB, devC);

        // place the chunk back to the host
        errchk(cudaMemcpyAsync(c + i, devC, N * sizeof(int), cudaMemcpyDeviceToHost, stream));
    }

    // Because our memcpys and kernel calls were async, we need to make sure
    // that everything is finished before freeing the memory
    errchk(cudaStreamSynchronize(stream));

    // free dev memory
    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    // free host memory using cudaFreeHost
    cudaFreeHost(a);
    cudaFreeHost(b);
    cudaFreeHost(c);

    // Destroy stream
    cudaStreamDestroy(stream);

    // measure time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    cout << "GPU time: " << elapsedTime << "ms\n";

    // destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
