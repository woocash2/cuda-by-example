#include "../cuda_errchk.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

using namespace std;


typedef chrono::high_resolution_clock Clock;

const int THREADS = 1024;

__global__ void sumCUDA(int * a, int * b, int * c, int n) {
    int i = blockIdx.x * THREADS + threadIdx.x;
    // warto sprawdzić jeśli ktoś próbowałby wywołać kernel z zbyt dużymi blockDim i threadDim
    if (i < n)
        c[i] = a[i] + b[i];
}

int ceil(int x, int y) {
    return (x + y - 1) / y;
}

int main() {

    int n; cin >> n;
    vector<int> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 1000; b[i] = rand() % 1000;
    }
    
    int * devA, * devB, * devC;
    errchk(cudaMalloc(&devA, n * sizeof(int)));
    errchk(cudaMalloc(&devB, n * sizeof(int)));
    errchk(cudaMalloc(&devC, n * sizeof(int)));
    
    errchk(cudaMemcpy(devA, a.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    errchk(cudaMemcpy(devB, b.data(), n * sizeof(int), cudaMemcpyHostToDevice));

    // Pomiar czasu przed rozpoczęciem pracy na GPU
    Clock::time_point before = Clock::now();

    sumCUDA<<<ceil(n, THREADS), THREADS>>>(devA, devB, devC, n);
    errchk(cudaDeviceSynchronize());

    // Pomiar czasu po wykonaniu pracy na GPU
    Clock::time_point after = Clock::now();

    // Liczba milisekund
    int time = chrono::duration_cast<chrono::milliseconds>(after - before).count();
    cout << "Czas pracy na GPU: " << time << "ms" << endl;

    errchk(cudaMemcpy(c.data(), devC, n * sizeof(int), cudaMemcpyDeviceToHost));

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

    for (int i = 0; i < n; i++) {
        if (c[i] != a[i] + b[i]) {
            cout << "Wrong answer " << i << " " << a[i] << " " << b[i] << " " << c[i];
            exit(1);
        }
    }

    cout << "OK";
    return 0;
}