#include "../common/common.h"
//#include <__clang_cuda_builtin_vars.h>
#include <stdio.h>
#include <cuda_runtime.h>

/*
 * A simple example of nested kernel launches from the GPU. Each thread displays
 * its information when execution begins, and also diagnostics when the next
 * lowest nesting layer completes.
 */

__global__ void nestedHelloWorld(int const iSize, int iDepth)
{
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid,
           blockIdx.x);

    // condition to stop recursive execution
    if (iSize == 1) return;

    // reduce block size to halfi
    int nthreads = iSize >> 1;

    // thread 0 launches child grid recursively
    if(tid == 0 && nthreads > 0)
    {
        nestedHelloWorld<<<1, nthreads>>>(nthreads, ++iDepth);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

__global__ void nestedHelloWorldExercise(const int iSize, int iDepth, int iLimit){

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1) return;

    // exercise 3-9: implement a new kernel that can limit nesting levels to a given depth
    if (iDepth > iLimit) return;

    int size = iSize >> 1;
    int blocksize = size >> 1; // new blocksize
    int depth = iDepth + 1;

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);

    if (idx == 0){
        nestedHelloWorldExercise<<<grid, block>>>(size, depth, iLimit);
        printf("-------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv)
{
    int size = 32;
    int blocksize = 16;   // initial block size
    int igrid = 1;

    if(argc > 1)
    {
        igrid = atoi(argv[1]);
        size = igrid * blocksize;
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("%s Execution Configuration: grid %d block %d\n", argv[0], grid.x,
           block.x);

    printf("Execution nestedHelloWorld()\n");
    nestedHelloWorld<<<grid, block>>>(block.x, 0);
    CHECK(cudaDeviceSynchronize());

    // exercise 3-8
    // Implement a new kernel that uses methods illustrated in Figure 3-30.
    // Figure 3-30 illustrates the case where the first thread of the first block
    // generates the nested call with half block size.
    //
    // exercise 3-9
    // implement a new kernel that can limit nesting levels to a given depth.
    // add one more parameter this will be condition to stop recursive execution.
    printf("\n");
    printf("Execution nestedHelloWorldExercise()\n");
    nestedHelloWorldExercise<<<grid, block>>>(size, 0, 2);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}
