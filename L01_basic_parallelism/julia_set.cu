#include <iostream>
#include <vector>

#include "../cuda_errchk.h"
#include <cuda_runtime.h>
#include <cuComplex.h>

using namespace std;


const int DIM = 1024;
const int PREC = 200;
const float initRe = -0.8, initIm = 0.156;


__device__ int is_in_julia(cuComplex z, cuComplex c) {

    for (int i = 0; i < PREC; i++) {
        z = cuCaddf(cuCmulf(z, z), c);
        if (cuCabsf(z) > 1000.0)
            return 0;
    }
    return 1;
}

__device__ cuComplex get_complex(int x, int y) {
    float scale = 1.5;
    float jx = scale * (float)(DIM - 2 * x) / DIM;
    float jy = scale * (float)(DIM - 2 * y) / DIM;
    return make_cuComplex(jx, jy);
}

__global__ void img_gen_CUDA(int * img) {
    int x = blockIdx.x;
    int y = threadIdx.x;
    int offset = x + y * DIM;

    cuComplex z = get_complex(x, y);
    int flag = is_in_julia(z, make_cuComplex(initRe, initIm));

    img[3 * offset + 0] = flag * 255;
    img[3 * offset + 1] = 0;
    img[3 * offset + 2] = (1 - flag) * 255;
}


int main() {

    // przygotuj obrazek na device
    int * devImg;
    errchk(cudaMalloc(&devImg, DIM * 3 * DIM * sizeof(int)));

    // generuj obrazek
    img_gen_CUDA<<<DIM, DIM>>>(devImg);

    // synchronizuj device
    errchk(cudaDeviceSynchronize());

    // przenieś obrazek do hosta
    vector<int> image(DIM * 3 * DIM, 0);
    errchk(cudaMemcpy(image.data(), devImg, DIM * 3 * DIM * sizeof(int), cudaMemcpyDeviceToHost));

    // posprzątaj na device
    cudaFree(devImg);

 

    // wypisanie wyniku (kompatybilny z eog)
    cout << "P3" << endl << DIM << " " << DIM << endl << 255 << endl;
    for (int i = 0; i < DIM * 3 * DIM; i += 3) {
        cout << image[i] << " " << image[i + 1] << " " << image[i + 2] << endl;
    }
    return 0;
}
