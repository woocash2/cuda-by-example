#include "book.h"
#include "cpu_anim.h"
#include <iostream>

using namespace std;

const int DIM = 1024;


struct DataBlock {
    unsigned char * devBitmap;
    CPUAnimBitmap * bitmap;
};

void cleanup(DataBlock * d) {
    cudaFree(d->devBitmap);
}

__global__ void getFrameCUDA(unsigned char * frame, int timestamp) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float fx = x - DIM / 2;
    float fy = y - DIM / 2;
    float d = sqrtf(fx * fx + fy * fy);

    unsigned char grey = (unsigned char) (128.0f + 127.0f * cos(d / 10.0f - timestamp / 7.0f) / (d / 10.0f + 1.0f));

    frame[4 * offset + 0] = grey;
    frame[4 * offset + 1] = grey;
    frame[4 * offset + 2] = grey;
    frame[4 * offset + 3] = 255;
}

// data - zawiera obrazek do wypełnienia. Kernel getFrameCUDA wypełni
// data->devBitmap. 
// tej bitmapy do zwykłej, hostowej bitmapy. generateFrame jest wywoływane
// z kolejnymi timestamp = 0,1,2,3,...
void generateFrame(DataBlock * data, int timestamp) {
    getFrameCUDA<<<{DIM / 16, DIM / 16}, {16, 16}>>>(data->devBitmap, timestamp);
    HANDLE_ERROR(cudaDeviceSynchronize());
    HANDLE_ERROR(cudaMemcpy(data->bitmap->get_ptr(), data->devBitmap, data->bitmap->image_size(), cudaMemcpyDeviceToHost));
}

int main() {
    DataBlock data;
    CPUAnimBitmap bitmap(DIM, DIM, &data);
    data.bitmap = &bitmap;

    HANDLE_ERROR(cudaMalloc(&data.devBitmap, bitmap.image_size()));

    // do wygenerowania każdej klatki używa funkcji generateFrame
    // cleanup do posprzątania po każdej klatce
    bitmap.anim_and_exit((void(*)(void*, int))generateFrame, (void(*)(void*))cleanup);

    return 0;
}