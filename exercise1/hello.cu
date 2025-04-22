#include <stdio.h>

__global__ void helloWorld(void){
    printf("Hello World frorm GPU!\n");
}

void helloFromGPU(dim3 grid, dim3 block){
    helloWorld<<<grid, block>>>();
    cudaDeviceSynchronize();
}

void helloFromCPU(void){
    printf("Hello World from CPU!\n");
}
