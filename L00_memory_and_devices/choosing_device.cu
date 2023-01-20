#include "../cuda_errchk.h"
#include <iostream>
#include <vector>

#include <cuda_runtime.h>


using namespace std;

__global__ void kernelDB(double a, double b) {
    a = a + b;
}

int main() {
    cudaDeviceProp prop;
    int dev;

    errchk(cudaGetDevice(&dev));  // zwraca numer używanego urządzenia
    cout << "Device before: " << dev << endl;

    memset(&prop, 0, sizeof(cudaDeviceProp)); // zerowanie struktury

    // Operacje na doublach są dostępne od wersji urządzenia 1.3
    prop.major = 1;
    prop.minor = 3;

    errchk(cudaChooseDevice(&dev, &prop));  // zwraca nr urządzenia o własnościach NAJBLIŻSZYCH do podanych w prop

    errchk(cudaSetDevice(dev));  // używaj tego urządzenia do obliczeń

    kernelDB<<<1, 1>>>(1.2, 1.3);

    cout << "Device after: " << dev << endl;
    return 0;
}