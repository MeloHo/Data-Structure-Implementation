// Ref: https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter2/01_variable_addition_value.cu

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void gpuAdd(int d_a, int d_b, int* d_c) {
    *d_c = d_a + d_b;
}


int main() {
    int h_c;
    int* d_c;

    cudaMalloc((void**)&d_c, sizeof(int));

    gpuAdd<<<1, 1>>>(1, 4, d_c);

    cudaMemcpy(&h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_c);
    return 0; 
}