#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>

/*
 * This example demonstrates a simple vector sum on the GPU and on the host.
 * sumArraysOnGPU splits the work of the vector sum across CUDA threads on the
 * GPU. Only a single thread block is used in this small case, for simplicity.
 * sumArraysOnHost sequentially iterates through vector elements on the host.
 * This version of sumArrays adds host timers to measure GPU and CPU
 * performance.
 */

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1.0E-8;
    bool match = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            printf("Arrays do not match!\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i],
                   gpuRef[i], i);
            break;
        }
    }

    if (match) printf("Arrays match.\n\n");

    return;
}

void initialData(float *ip, int size)
{
    // generate different seed for random number
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i < size; i++)
    {
        ip[i] = (float)( rand() & 0xFF ) / 10.0f;
    }

    return;
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    for (int idx = 0; idx < N; idx++)
    {
        C[idx] = A[idx] + B[idx];
    }
}
__global__ void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N){
        C[i] = A[i] + B[i];
    }
}

__global__ void sumArraysOnGPU_cycling(float *A, float *B, float *C, const int N, const int offset){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < offset) {
      C[i] = A[i] + B[i];
      C[i + offset] = A[i + offset] + B[i + offset];
    }
}

__global__ void sumArraysOnGPU_neighbor(float *A, float *B, float *C, const int N, const int offset){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < offset){
        C[2*i] = A[2*i] + B[2*i];
        C[2*i+1] = A[2*i+1] + B[2*i+1];
    }
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n", argv[0]);

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of vectors
    int nElem = 1 << 24;
    printf("Vector size %d\n", nElem);

    // malloc host memory
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    double iStart, iElaps;

    // initialize data at host side
    iStart = seconds();
    initialData(h_A, nElem);
    initialData(h_B, nElem);
    iElaps = seconds() - iStart;
    printf("initialData Time elapsed %f sec\n", iElaps);
    memset(hostRef, 0, nBytes); // sets all bytes to zero.
    memset(gpuRef,  0, nBytes);

    // add vector at host side for result checks
    iStart = seconds();
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    iElaps = seconds() - iStart;
    printf("sumArraysOnHost Time elapsed %f sec\n", iElaps);

    // malloc device global memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((float**)&d_A, nBytes));
    CHECK(cudaMalloc((float**)&d_B, nBytes));
    CHECK(cudaMalloc((float**)&d_C, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    // exercise 2-1: compare the result with the execution configuration of
    //               block.x = 1023, 1024
    //
    // if block.x = 1023, blockDim.x = 16401
    // if block.x = 1024, blockDim.x = 16384
    // The main difference is the blockDim.x.
    //
    // 1. If there are more threads in a block, less thread blocks are needed.
    // less threads requires more thread blocks and this turns out to be
    // the reason for execution time, because only a fixed number of
    // thread blocks can run concurrently.
    //
    // 2. block.x = 1023 turns out to be waste of threads
    // because the last warp in every thread block will have a disabled
    // thread that performs no work because 1023 cannot be divided by the warp
    // size of 32.
    //
    //
    // More factors could be Occupancy due to warp size of 32..
    // and this is covered in Chapter 3.

    // invoke kernel at host side
    int iLen = 1023;
    dim3 block (iLen);
    dim3 grid  ((nElem + block.x - 1) / block.x); // truncated

    iStart = seconds();
    sumArraysOnGPU<<<grid, block>>>(d_A, d_B, d_C, nElem);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);

    //
    // exercise 2-2: set block.x = 256, and let each thread handle two elements
    // here we test two more functions: sumArraysOnGPU_cycle(), sumArraysOnGPU_neighbor()
    // sumArraysOnGPU_cycle() allows each thread handle two elements in cycling manner
    // sumArraysOnGPU_neighbor() allows each thread handle two elements consecutively.
    //
    // Having two threads handle two elements turns out to be at least 30% faster.
    // Namely it is similar to loop unrolling, less overheads.
    iLen = 256;
    block = (iLen);
    grid = ((nElem/2 + block.x - 1) / block.x); // grid dimension should be half, too.

    iStart = seconds();
    sumArraysOnGPU_cycling<<<grid, block>>>(d_A, d_B, d_C, nElem, nElem >> 1);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU_cycling <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);


    iStart = seconds();
    sumArraysOnGPU_neighbor<<<grid, block>>>(d_A, d_B, d_C, nElem, nElem >> 1);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("sumArraysOnGPU_neighbor <<<  %d, %d  >>>  Time elapsed %f sec\n", grid.x,
           block.x, iElaps);


    // check kernel error
    CHECK(cudaGetLastError()) ;

    // copy kernel result back to host side
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    checkResult(hostRef, gpuRef, nElem);

    // free device global memory
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    // free host memory
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);

    return(0);
}
