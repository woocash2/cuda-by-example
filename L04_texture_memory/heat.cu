#include "../common/book.h"
#include "../common/cpu_anim.h"
#include <iostream>

using namespace std;

#define DIM 1024
#define MIN_TEMP 0.0001f
#define MAX_TEMP 1.0f
#define HEAT_RATE 0.25f


struct DataBlock {
    unsigned char * devBmp;
    float * devInpGrid;
    float * devOutGrid;
    float * devConstGrid;
    CPUAnimBitmap * bmp;
    cudaEvent_t start, stop;
    float totalTime;
    int frames;
};


texture<float> texConstGrid;
texture<float> texInpGrid;
texture<float> texOutGrid;


void initConstGrid(float * constGrid) {
    for (int i = 0; i < DIM * DIM; i++) {
        constGrid[i] = 0;
        int x = i % DIM;
        int y = i / DIM;
        if (x > 300 && x < 600 && y > 310 && y < 601)
            constGrid[i] = MAX_TEMP;
    }
    constGrid[DIM*100+100] = (MAX_TEMP + MIN_TEMP)/2;
    constGrid[DIM*700+100] = MIN_TEMP;
    constGrid[DIM*300+300] = MIN_TEMP;
    constGrid[DIM*200+700] = MIN_TEMP;
    for (int y=800; y<900; y++) {
        for (int x=400; x<500; x++) {
            constGrid[x+y*DIM] = MIN_TEMP;
        }
    }
}


void initInpGrid(float * inpGrid) {
    for (int y=800; y<DIM; y++) {
        for (int x=0; x<200; x++) {
            inpGrid[x+y*DIM] = MAX_TEMP;
        }
    }
}


__global__ void copyConstCUDA(float * inpGrid) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float c = tex1Dfetch(texConstGrid, offset);
    if (c != 0)
        inpGrid[offset] = c;
}


__global__ void gridBlendCUDA(float * dst, bool dstOut) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    int lef = offset - 1;
    int rig = offset + 1;
    if (x == 0)   lef++;
    if (x == DIM-1) rig--; 

    int up = offset - DIM;
    int dn = offset + DIM;
    if (y == 0)   up += DIM;
    if (y == DIM-1) dn -= DIM;

    float l, r, u, d, c;
    if (dstOut) {
        l = tex1Dfetch(texInpGrid, lef);
        r = tex1Dfetch(texInpGrid, rig);
        u = tex1Dfetch(texInpGrid, up);
        d = tex1Dfetch(texInpGrid, dn);
        c = tex1Dfetch(texInpGrid, offset);
    }
    else {
        l = tex1Dfetch(texOutGrid, lef);
        r = tex1Dfetch(texOutGrid, rig);
        u = tex1Dfetch(texOutGrid, up);
        d = tex1Dfetch(texOutGrid, dn);
        c = tex1Dfetch(texOutGrid, offset);
    }

    dst[offset] = c + HEAT_RATE * (l + r + u + d - 4 * c);
}


void makeFrame(DataBlock * data, int tick) {
    HANDLE_ERROR(cudaEventRecord(data->start, 0));

    dim3 blocks(DIM / 16, DIM / 16);
    dim3 threads(16, 16);

    volatile bool dstOut = true;
    for (int i = 0; i < 90; i++) {
        float * inp, * out;
        if (dstOut) {
            inp = data->devInpGrid;
            out = data->devOutGrid;
        }
        else {
            out = data->devInpGrid;
            inp = data->devOutGrid;
        }
        copyConstCUDA<<<blocks, threads>>>(inp);
        gridBlendCUDA<<<blocks, threads>>>(out, dstOut);
        dstOut = !dstOut;
    }

    float_to_color<<<blocks, threads>>>(data->devBmp, data->devInpGrid);

    HANDLE_ERROR(cudaMemcpy(data->bmp->get_ptr(), data->devBmp, data->bmp->image_size(), cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaEventRecord(data->stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(data->stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, data->start, data->stop));

    data->totalTime += elapsedTime;
    ++data->frames;

    if (data->frames % 10 == 0) {
        cout << "Average time per frame: " << data->totalTime / (float)data->frames << "ms" << endl;
    }
}


void frameCleanup(DataBlock * data) {
    cudaUnbindTexture(texInpGrid);
    cudaUnbindTexture(texOutGrid);
    cudaUnbindTexture(texConstGrid);

    cudaFree(data->devInpGrid);
    cudaFree(data->devOutGrid);
    cudaFree(data->devConstGrid);

    HANDLE_ERROR(cudaEventDestroy(data->start));
    HANDLE_ERROR(cudaEventDestroy(data->stop));
}


int main() {

    // init DataBlock struct
    DataBlock data;
    CPUAnimBitmap bmp(DIM, DIM, &data);
    data.bmp = &bmp;
    data.totalTime = 0;
    data.frames = 0;
    HANDLE_ERROR(cudaEventCreate(&data.start));
    HANDLE_ERROR(cudaEventCreate(&data.stop));


    // init device grids and bitmap
    // bmp.image_size() is proper because float == 4 * char
    HANDLE_ERROR(cudaMalloc(&data.devBmp, bmp.image_size()));
    HANDLE_ERROR(cudaMalloc(&data.devConstGrid, bmp.image_size()));
    HANDLE_ERROR(cudaMalloc(&data.devInpGrid, bmp.image_size()));
    HANDLE_ERROR(cudaMalloc(&data.devOutGrid, bmp.image_size()));


    // bind the texture to the memory references just allocated
    HANDLE_ERROR(cudaBindTexture(NULL, texConstGrid, data.devConstGrid, bmp.image_size()));
    HANDLE_ERROR(cudaBindTexture(NULL, texInpGrid, data.devInpGrid, bmp.image_size()));
    HANDLE_ERROR(cudaBindTexture(NULL, texOutGrid, data.devOutGrid, bmp.image_size()));

    
    // make const grid and move it into the device
    float * temp = (float *) malloc(DIM * DIM * sizeof(float));
    initConstGrid(temp);
    cudaMemcpy(data.devConstGrid, temp, bmp.image_size(), cudaMemcpyHostToDevice);
    initInpGrid(temp);
    cudaMemcpy(data.devInpGrid, temp, bmp.image_size(), cudaMemcpyHostToDevice);
    free(temp);


    bmp.anim_and_exit((void(*)(void*, int))makeFrame, (void(*)(void*))frameCleanup);
    return 0;
}