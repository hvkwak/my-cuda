#include <stdio.h>

__global__ void helloWorld(void){

    // exercise 1-6: let it print only for the 5-th thread.
    if (threadIdx.x == 4){
        printf("Hello World from GPU thread 5!\n");
    }
    //printf("Hello World from GPU thread %d!\n", threadIdx.x);
}

void helloFromGPU(dim3 grid, dim3 block){
    helloWorld<<<grid, block>>>();
    //cudaDeviceSynchronize();
}

void helloFromCPU(void){
    printf("Hello World from CPU!\n");
}
