#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using a statically declared global variable (devData) to store
 * a floating-point value on the device.
 */

__device__ float devData;

__device__ float devArray[5];

__global__ void checkGlobalVariable()
{
    // display the original value
    printf("Device: the value of the global variable is %f\n", devData);

    // alter the value
    devData += 2.0f;
}

__global__ void checkGlobalArray(){

    int tid = threadIdx.x;
    if (tid < 5){

        // modify the value
        devArray[tid] = devArray[tid]*(float)tid;
    }
}

__global__ void printGlobalArray(){

    int tid = threadIdx.x;
    if (tid < 5){
        // display the value
        printf("Device: the i-th value of the global array is %f\n", devArray[tid]);
    }
}

int main(void)
{

    // set up device - no need: default is dev = 0;
    // int dev = 0;
    // cudaSetDevice(dev);

    // initialize the global variable
    float value = 3.14f;
    CHECK(cudaMemcpyToSymbol(devData, &value, sizeof(float)));
    printf("Host:   copied %f to the global variable\n", value);

    // invoke the kernel
    checkGlobalVariable<<<1, 1>>>();

    // copy the global variable back to the host
    CHECK(cudaMemcpyFromSymbol(&value, devData, sizeof(float)));
    printf("Host:   the value changed by the kernel to %f\n", value);

    // exercise 4-1:
    // 1. initialize the global array with the same value of 3.14 with a size
    //    of five elements
    // 2. Invoke a kernel with five threads to..
    //    modify the kernel to let each thread change the value of the array
    //    element with the same index as the thread index AND
    //    let the value be multiplied with the thread index.
    printf("exercise 4-1*******************\n");
    float valueArray[5] = {3.14f, 3.14f, 3.14f, 3.14f, 3.14f};
    CHECK(cudaMemcpyToSymbol(devArray, &valueArray, sizeof(float)*5)); // To symbol "devArray"
    for(int i = 0; i < 5; i++){
        printf("Host:   copied %f to the global array \n", valueArray[i]);
    }

    checkGlobalArray<<<1, 5>>>();

    CHECK(cudaMemcpyFromSymbol(&valueArray, devArray, sizeof(float)*5)); // From symbol "devArray"
    for(int i = 0; i < 5; i++){
        printf("Host:   the value changed by the kernel to %f\n", valueArray[i]);
    }


    // exercise 4-2: replace cudaMemcpyToSymbol() and cudaMemcpyFromSymbol()
    //               with cudaMemcpy() mittels cudaGetSymbolAddress()
    printf("exercise 4-2*******************\n");
    float *dptr = NULL; // pointer to the first element of devArray
    CHECK(cudaGetSymbolAddress((void **)&dptr, devArray));
    CHECK(cudaMemcpy(dptr, &valueArray, sizeof(float)*5, cudaMemcpyHostToDevice));
    printGlobalArray<<<1, 5>>>();


    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
