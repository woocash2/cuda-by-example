#include "../cuda_errchk.h"

int main() {
    cudaDeviceProp prop;
    int count;
    errchk(cudaGetDeviceCount(&count));
    for (int i = 0; i < count; i++) {
        errchk(cudaGetDeviceProperties(&prop, i));
        printf("--- General info about device %d ---\n", i);
        printf("Name %s\n", prop.name);
        printf("Compute capability %d.%d\n", prop.major, prop.minor);
        printf("Clock rate %d\n", prop.clockRate);
        printf("Total global mem %ld\n", prop.totalGlobalMem);
        printf("Total constant mem %ld\n", prop.totalConstMem);
        printf("Multiprocessor count %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp %ld\n", prop.sharedMemPerMultiprocessor);
        printf("Threads in warp %d\n", prop.warpSize);
        printf("Num of threads per block %d\n\n", prop.maxThreadsPerBlock);
    }

    return 0;
}