#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using namespace std;

__global__ void matmulCUDA(int * a, int * b, int * c, int n) {
    int i = blockIdx.x;
    int j = threadIdx.x;

    c[n * i + j] = 0;
    for (int k = 0; k < n; k++) {
        c[n * i + j] += a[n * i + k] * b[n * k + j];
    }
}

int main() {

    int n; cin >> n;
    vector<int> A(n * n), B(n * n), C(n * n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> A[n * i + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cin >> B[n * i + j];
        }
    }

    int * devA, *devB, *devC;
    cudaMalloc(&devA, n * n * sizeof(int));
    cudaMalloc(&devB, n * n * sizeof(int));
    cudaMalloc(&devC, n * n * sizeof(int));

    cudaMemcpy(devA, A.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devB, B.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);

    matmulCUDA<<<n, n>>>(devA, devB, devC, n);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), devC, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << C[n * i + j] << " ";
        }
        cout << endl;
    }

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    return 0;
}