#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include "cuda.h"

#include <iostream>
const int DIM = 1024;
const float PI = 3.1415926535897932f;
const float period = 128.0f;


__global__ void squaresCUDA(unsigned char * bmp) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = tx + blockIdx.x * blockDim.x;
    int y = ty + blockIdx.y * blockDim.y;

    int offset = x + y * blockDim.x * gridDim.x;

    __shared__ float imgpart[16][16];

    imgpart[tx][ty] = 255.0f * (sinf(x * 2.0f * PI / period) + 1.0f) *
                               (sinf(y * 2.0f * PI / period) + 1.0f) / 4.0f;
    __syncthreads();

    bmp[4 * offset + 0] = 0;
    // wymaga __syncthreads() bo korzystamy z "nie swojej" komórki pamięci shared
    // gdyby zamiast (15 - tx) było tx <ty>, nie musielibyśmy tego robić 
    bmp[4 * offset + 1] = imgpart[15 - tx][15 - ty];
    bmp[4 * offset + 2] = 0;
    bmp[4 * offset + 3] = 255;
}


int main() {
    CPUBitmap bitmap(DIM, DIM);
    unsigned char * devBitmap;
    HANDLE_ERROR(cudaMalloc(&devBitmap, bitmap.image_size()));

    squaresCUDA<<<dim3(DIM / 16, DIM / 16), dim3(16, 16)>>>(devBitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), devBitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

    cudaFree(devBitmap);

    bitmap.display_and_exit();
}
