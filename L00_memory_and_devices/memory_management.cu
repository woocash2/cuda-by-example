#include "../cuda_errchk.h"
#include <iostream>
#include <vector>

#include <cuda_runtime.h>


using namespace std;

// global oznacza wykonywany na device ale możliwy do wołania z hosta
__global__ void kernel(int a, int b, int * c) {  // c będzie wskaźnikiem do pamięci na urządzeniu
    *c = a + b;
}

int main() {
    
    int c = 0;

    int * devC;                                                         // wskaźnik docelowo do pamięci na urządzeniu
    errchk(cudaMalloc(&devC, sizeof(int)));                             // alokacja na urządzeniu

    kernel<<<1, 1>>>(5, 10, devC);                                      // wywołanie kernela


    errchk(cudaMemcpy(&c, devC, sizeof(int), cudaMemcpyDeviceToHost));  // odzyskanie wyniku do pamięci hosta

    cout << c << endl;

    cudaFree(devC);                                                     // tak jak po zwykłym malloc, trzeba zwolnić
}