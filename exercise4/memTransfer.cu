#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using CUDA's memory copy API to transfer data to and from the
 * device. In this case, cudaMalloc is used to allocate memory on the GPU and
 * cudaMemcpy is used to transfer the contents of host memory to an array
 * allocated using cudaMalloc.
 */

int main(int argc, char **argv)
{
    if (argc < 2){
        printf("Usage: %s <integer>\n", argv[0]);
        return 1;
    }

    unsigned int value = atoi(argv[1]);

    // set up device
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    // memory size
    unsigned int isize = 1 << value;
    unsigned int nbytes = isize * sizeof(float);

    double Start, Elaps, Allocate, Deallocate;

    // get device information
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    // printf("%s    starting at ", argv[0]); // too many strirngs
    printf("device %d: %s memory size %d nbyte %5.2fMB ", dev,
           deviceProp.name, isize, nbytes / (1024.0f * 1024.0f));

    // allocate the host memory
    Allocate = seconds();
    float *h_a = (float *)malloc(nbytes);
    Allocate = seconds() - Allocate;

    // allocate the device memory
    float *d_a;
    CHECK(cudaMalloc((float **)&d_a, nbytes));

    // initialize the host memory
    for(unsigned int i = 0; i < isize; i++) h_a[i] = 0.5f;


    Start = seconds();
    // transfer data from the host to the device
    CHECK(cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice));

    // transfer data from the device to the host
    CHECK(cudaMemcpy(h_a, d_a, nbytes, cudaMemcpyDeviceToHost));
    Elaps = seconds() - Start;

    // free memory
    Deallocate = seconds();
    CHECK(cudaFree(d_a));
    Deallocate = seconds() - Deallocate;
    free(h_a);

    printf("gpu memTransfer elapsed %f sec, Alloc. %f sec, Dealloc. %f sec\n", Elaps, Allocate, Deallocate);

    // reset device
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
