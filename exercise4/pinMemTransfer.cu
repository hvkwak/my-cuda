#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

/*
 * An example of using CUDA's memory copy API to transfer data to and from the
 * device. In this case, cudaMalloc is used to allocate memory on the GPU and
 * cudaMemcpy is used to transfer the contents of host memory to an array
 * allocated using cudaMalloc. Host memory is allocated using cudaMallocHost to
 * create a page-locked host array.
 */

int main(int argc, char **argv)
{
    // exercise 4-4: using the same examples, the performance of pinned memory
    // allocation generally better than pageable memory.
    // if (argc < 2){
    //     printf("Usage: %s <integer>\n", argv[0]);
    //     return 1;
    // }
    // unsigned int value = atoi(argv[1]);

    for (unsigned int value = 19; value < 26; value++){
      // 19 -> 2MB
      // 25 -> 128MB
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

      if (!deviceProp.canMapHostMemory) {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
      }

      // printf("%s starting at ", argv[0]); // too many strings
      printf("device %d: %s memory size %d, %5.2fMB canMap %d ", dev,
             deviceProp.name, isize, nbytes / (1024.0f * 1024.0f),
             deviceProp
                 .canMapHostMemory); // TODO: check what this MapHostMemory is

      // allocate pinned host memory
      Start = seconds();
      float *h_a;
      CHECK(cudaMallocHost((float **)&h_a, nbytes));
      Allocate = seconds() - Start;

      // allocate device memory
      float *d_a;
      CHECK(cudaMalloc((float **)&d_a, nbytes));

      // initialize host memory
      memset(h_a, 0, nbytes);

      for (int i = 0; i < isize; i++)
        h_a[i] = 100.10f;

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
      CHECK(cudaFreeHost(h_a));

      printf(
          "gpu pinMemTransfer elapsed %f sec, Alloc. %f sec, Dealloc. %f sec\n",
          Elaps, Allocate, Deallocate);

      // reset device
      CHECK(cudaDeviceReset());
    }
    return EXIT_SUCCESS;
}
