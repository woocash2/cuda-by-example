#include "../common/book.h"
#include <iostream>
#include <vector>

using namespace std;
const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;


// dot(val(0, *), val(1, *)) powinien zwrócić podwojoną sumę kwadratów n kolejnych liczb naturalnych
__device__ long long val(int arr, int i) {
    return (1 + arr) * i;
}

__global__ void dotProdCUDA(long long * c, long long n) {

    __shared__ long long partial[threadsPerBlock];

    int t = threadIdx.x;
    int i = t + blockIdx.x * blockDim.x;

    long long temp = 0;

    while (i < n) {
        temp += val(0, i + 1) * val(1, i + 1);
        i += blockDim.x * gridDim.x;
    }

    partial[t] = temp;

    // wszyscy skończyli liczyć częściowe
    __syncthreads();

    int j = blockDim.x;
    while (j /= 2) {
        if (t < j)
            partial[t] += partial[t + j];
        // wszyscy czekają dopóki nowa podtablica nie będzie wypełniona
        // nie można tego wołać w ciele if (t < j) bo __syncthreads() czeka
        // na dokładnie wszystkie wątki, a do ifa nie wchodzą wszystkie.
        __syncthreads();
    }

    if (t == 0)
        c[blockIdx.x] = partial[0];
}


int main() {
    long long n; cin >> n;
    long long ans = n * (n + 1) * (2 * n + 1) / 3;

    int blocks = min((long long)blocksPerGrid, (n + threadsPerBlock - 1) / threadsPerBlock);

    long long * cDev;
    cudaMalloc(&cDev, blocks * sizeof(long long));

    dotProdCUDA<<<blocks, threadsPerBlock>>>(cDev, n);

    vector<long long> c(blocks);
    cudaMemcpy(c.data(), cDev, blocks * sizeof(long long), cudaMemcpyDeviceToHost);

    cudaFree(cDev);

    long long res = 0;
    for (auto x : c)
        res += x;

    if (res == ans)
        cout << "OK";
    else
        cout << "Wrong answer: your: " << res << " correct: " << ans;
}