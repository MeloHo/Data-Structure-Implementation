// Ref:https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter2/02_variable_addition_reference.cu

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
    *d_c = *d_a + *d_b;
}

int main() {
    int h_a, h_b, h_c;
    int *d_a, *d_b, *d_c;

    h_a = 1;
    h_b = 4;

    cudaMalloc((void**)&d_a, sizeof(int));
    cudaMalloc((void**)&d_b, sizeof(int));
    cudaMalloc((void**)&d_c, sizeof(int));

    cudaMemcpy(d_a, &h_a, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &h_b, sizeof(int), cudaMemcpyHostToDevice);

    gpuAdd<<<1, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_a);
    cudaFree(d_a);

    return 0;
}