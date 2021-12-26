// Ref:https://github.com/PacktPublishing/Hands-On-GPU-Accelerated-Computer-Vision-with-OpenCV-and-CUDA/blob/master/Chapter2/03_thread_execution_example.cu

#include <iostream>
#include <stdio.h>

__global__ void myfirstkernel(void) {
    printf("Hello! I'm thread in block: %d\n", blockIdx.x);
}


int main() {
    myfirstkernel<<<16, 1>>>();

    cudaDeviceSynchronize();
    printf("All threads are finished.\n");

    return 0;
}