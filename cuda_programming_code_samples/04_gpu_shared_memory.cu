// Ref:https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter3/04_gpu_shared_memory.cu

#include <stdio.h>

__global__ void gpu_shared_memory(float* d_a) {
    int i, index = threadIdx.x;
    float average, sum = 0.0f;

    // Define shared memory
    __shared__ float sh_arr[10];

    sh_add[index] = d_a[index];

    __syncthreads();

    for (int i = 0; i <= index; i++) {
        sum += sh_arr[i];
    }

    average = sum / (index + 1.0f);

    d_a[index] = average;
    sh_arr[index] = average;
}

int main() {
    float h_a[10];
    float* d_a;

    for (int i = 0; i < 10; i++) {
        h_a[i] = i;
    }

    cudaMalloc((void**)&d_a, 10 * sizeof(float));
    cudaMemcpy((void*)d_a, (void*)h_a, 10 * sizeof(float), cudaMemcpyHostToDevice);
    gpu_shared_memory<<<1, 10>>>(d_a);

    cudaMemcpy((void*)h_a, (void*)d_a, 10 * sizeof(float), cudaMemcpyDeviceToHost);


    return 0;
}