#include <iostream>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel to add two vectors
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int n = 1000000; // Size of vectors
    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // Host vectors
    float *h_a, *h_b, *h_c;
    h_a = new float[n];
    h_b = new float[n];
    h_c = new float[n];

    // Initialize host vectors
    for (int i = 0; i < n; ++i) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device vectors
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n * sizeof(float));
    cudaMalloc((void **)&d_b, n * sizeof(float));
    cudaMalloc((void **)&d_c, n * sizeof(float));

    // Copy host vectors to device
    cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    // Copy result back to host
    cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify the result
    bool success = true;
    for (int i = 0; i < n; ++i) {
        if (h_c[i] != h_a[i] + h_b[i]) {
            std::cout << "Error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << std::endl;
            success = false;
            break;
        }
    }

    if (success)
        std::cout << "Vectors added successfully!" << std::endl;

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
