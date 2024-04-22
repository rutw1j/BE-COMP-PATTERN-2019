/*
To run CUDA in colab

!pip install git+https://github.com/andreinechaev/nvcc4jupyter.git
%load_ext nvcc4jupyter

%%cuda
*/


#include <iostream>
#include <cuda_runtime.h>

using namespace std;

#define TILE_WIDTH 16

// CUDA kernel for matrix multiplication
__global__ void matrixMul(const float *a, const float *b, float *c, int width) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    for (int k = 0; k < width; ++k) {
        float elementA = a[row * width + k];
        float elementB = b[k * width + col];
        sum += elementA * elementB;
    }

    c[row * width + col] = sum;
}

int main() {
    const int width = 1024; // Matrix width
    const int size = width * width * sizeof(float);

    // Host matrices
    float *h_a = new float[width * width];
    float *h_b = new float[width * width];
    float *h_c = new float[width * width];

    // Initialize host matrices
    for (int i = 0; i < width * width; ++i) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Device matrices
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy host matrices to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch kernel
    matrixMul<<<gridDim, blockDim>>>(d_a, d_b, d_c, width);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the result (optional)
    // for (int i = 0; i < width * width; ++i) {
    //     std::cout << h_c[i] << " ";
    //     if ((i + 1) % width == 0)
    //         std::cout << std::endl;
    // }

    // Free device memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    // Free host memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;

    return 0;
}
