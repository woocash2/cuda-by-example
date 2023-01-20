#include "../common/book.h"
#include "../common/cpu_bitmap.h"
#include <iostream>
using namespace std;

#define DIM 1024
#define SPHERES 20
#define INF 2e10f
#define rnd(x) ((x) * rand() / RAND_MAX)

struct Sphere {
    float r, g, b;
    float radius;
    float x, y, z;

    __device__ float hit(float ox, float oy, float * light_fraction) {
        float dx = ox - x;
        float dy = oy - y;

        if (dx * dx + dy * dy < radius * radius) {
            float dz = sqrtf(radius * radius - (dx * dx + dy * dy));
            *light_fraction = dz / radius;
            return dz + z;
        }
        return -INF;
    }
};


// We optimize using constant GPU memory for the data which is heavily used and
// not modified
__constant__ Sphere devSpheres[SPHERES];


__global__ void rayTraceCUDA(unsigned char * bmp) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int offset = x + y * blockDim.x * gridDim.x;

    float ox = x - DIM / 2;
    float oy = y - DIM / 2;

    float r = 0, g = 0, b = 0;
    float maxz = -INF;

    for (int i = 0; i < SPHERES; i++) {
        float light_factor = 0;
        float dist = devSpheres[i].hit(ox, oy, &light_factor);
        if (dist > maxz) {
            r = devSpheres[i].r * light_factor;
            g = devSpheres[i].g * light_factor;
            b = devSpheres[i].b * light_factor;
            maxz = dist;
        }
    }

    bmp[4 * offset + 0] = (unsigned char) (r * 255);
    bmp[4 * offset + 1] = (unsigned char) (g * 255);
    bmp[4 * offset + 2] = (unsigned char) (b * 255);
    bmp[4 * offset + 3] = 255;
}


int main() {
    srand(42);

    // creating events and starting the start event
    cudaEvent_t start, stop;
    HANDLE_ERROR(cudaEventCreate(&start));
    HANDLE_ERROR(cudaEventCreate(&stop));
    HANDLE_ERROR(cudaEventRecord(start, 0));


    // creating random SPHERES spheres
    Sphere * spheres = (Sphere*)malloc(SPHERES * sizeof(Sphere));
    for (int i = 0; i < SPHERES; i++) {
        spheres[i].r = rnd(1.0f);
        spheres[i].g = rnd(1.0f);
        spheres[i].b = rnd(1.0f);

        spheres[i].x = rnd(1000.0f) - 500.0f;
        spheres[i].y = rnd(1000.0f) - 500.0f;
        spheres[i].z = rnd(1000.0f) - 500.0f;

        spheres[i].radius = rnd(100.0f) + 20.0f;
    }

    // copying the host spheres to constant memory on the GPU
    HANDLE_ERROR(cudaMemcpyToSymbol(devSpheres, spheres, SPHERES * sizeof(Sphere)));


    // --- standard CUDA proceedings ---

    // Allocating bitmap and dev bitmap
    CPUBitmap bmp(DIM, DIM);
    unsigned char * devBmp;
    HANDLE_ERROR(cudaMalloc(&devBmp, bmp.image_size()));

    // copying spheres from host to device
    cudaMemcpy(devSpheres, spheres, SPHERES * sizeof(Sphere), cudaMemcpyHostToDevice);

    // kernel call to ray trace
    rayTraceCUDA<<<{DIM / 16, DIM / 16}, {16, 16}>>>(devBmp);

    // restore the host bitmap from the device bitmap
    HANDLE_ERROR(cudaMemcpy(bmp.get_ptr(), devBmp, bmp.image_size(), cudaMemcpyDeviceToHost));

    // cleanup on both host and device
    // no need to worry about freeing __constant__ memory
    free(spheres);
    cudaFree(devBmp);


    // get the time of the computation
    HANDLE_ERROR(cudaEventRecord(stop, 0));
    HANDLE_ERROR(cudaEventSynchronize(stop));
    float elapsedTime;
    HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
    cout << "Took time: " << elapsedTime << "ms\n";

    // destroy events
    HANDLE_ERROR(cudaEventDestroy(start));
    HANDLE_ERROR(cudaEventDestroy(stop));


    // display the results
    bmp.display_and_exit();

    return 0;
}