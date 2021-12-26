// Ref: https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter2/05_vector_addition_gpu.cu
#include "stdio.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5

__global__ void gpuAdd(int* d_a, int* d_b, int* d_c) {
    int tid = blockIdx.x;
    if (tid < N)
        d_c[tid] = d_a[tid] + d_b[tid];
}


int main() {
    int h_a[N], h_b[N], h_c[N];
    int *d_a, *d_b, *d_c;

    cudaMelloc((void**)&d_a, N * sizeof(int));
    cudaMelloc((void**)&d_b, N * sizeof(int));
    cudaMelloc((void**)&d_c, N * sizeof(int));
    
    for (int i = 0; i < N; i++) {
        h_a[i] = 2*i*i;
        h_b[i] = i;
    }

    cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, N * sizeof(int), cudaMemcpyHostToDevice);

    gpuAdd<<<N, 1>>>(d_a, d_b, d_c);

    cudaMemcpy(h_c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}