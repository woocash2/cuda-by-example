#include <iostream>
#include <cuda_runtime.h>
#include "cuda_errchk.h"

using namespace std;


const int SIZE = 1024 * 1024 * 100; // 100MB


float mallocTest(bool toDevice) {
    cudaEvent_t start, stop;
    errchk(cudaEventCreate(&start));
    errchk(cudaEventCreate(&stop));
    errchk(cudaEventRecord(start, 0));

    unsigned char * a = (unsigned char *) malloc(SIZE);
    unsigned char * devA;
    errchk(cudaMalloc(&devA, SIZE));


    for (int i = 0; i < 100; i++) {
        if (toDevice)
            errchk(cudaMemcpy(devA, a, SIZE, cudaMemcpyHostToDevice));
        else
            errchk(cudaMemcpy(a, devA, SIZE, cudaMemcpyDeviceToHost));
    }

    free(a);
    cudaFree(devA);

    errchk(cudaEventRecord(stop, 0));
    errchk(cudaEventSynchronize(stop));
    float elapsedTime;
    errchk(cudaEventElapsedTime(&elapsedTime, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}


float cudaHostAallocTest(bool toDevice) {
    cudaEvent_t start, stop;
    errchk(cudaEventCreate(&start));
    errchk(cudaEventCreate(&stop));
    errchk(cudaEventRecord(start, 0));

    unsigned char * a;
    errchk(cudaHostAlloc(&a, SIZE, cudaHostAllocDefault));
    unsigned char * devA;
    errchk(cudaMalloc(&devA, SIZE));


    for (int i = 0; i < 100; i++) {
        if (toDevice)
            errchk(cudaMemcpy(devA, a, SIZE, cudaMemcpyHostToDevice));
        else
            errchk(cudaMemcpy(a, devA, SIZE, cudaMemcpyDeviceToHost));
    }

    cudaFreeHost(a);
    cudaFree(devA);

    errchk(cudaEventRecord(stop, 0));
    errchk(cudaEventSynchronize(stop));
    float elapsedTime;
    errchk(cudaEventElapsedTime(&elapsedTime, start, stop));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsedTime;
}


int main() {

    cout << "GPU times of copying 100MB:" << endl;
    cout << "C malloc host to device: " << mallocTest(true) << "ms" << endl;
    cout << "C malloc device to host: " << mallocTest(false) << "ms" << endl;
    cout << "cudaHostAlloc host to device: " << cudaHostAallocTest(true) << "ms" << endl;
    cout << "cudaHostAlloc device to host: " << cudaHostAallocTest(false) << "ms" << endl;


    return 0;
}