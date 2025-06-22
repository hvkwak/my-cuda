#include "../common/common.h"
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <stdio.h>



float recursiveReduce(float *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

__global__ void reduceCompleteUnrollWarps8(float *g_idata, float *g_odata,
        unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // unrolling 8
    if (idx + 7 * blockDim.x < n)
    {
        float a1 = g_idata[idx];
        float a2 = g_idata[idx + blockDim.x];
        float a3 = g_idata[idx + 2 * blockDim.x];
        float a4 = g_idata[idx + 3 * blockDim.x];
        float b1 = g_idata[idx + 4 * blockDim.x];
        float b2 = g_idata[idx + 5 * blockDim.x];
        float b3 = g_idata[idx + 6 * blockDim.x];
        float b4 = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }

    __syncthreads();

    // in-place reduction and complete unroll
    if (blockDim.x >= 1024 && tid < 512) idata[tid] += idata[tid + 512];

    __syncthreads();

    if (blockDim.x >= 512 && tid < 256) idata[tid] += idata[tid + 256];

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128) idata[tid] += idata[tid + 128];

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64) idata[tid] += idata[tid + 64];

    __syncthreads();

    // unrolling warp
    if (tid < 32)
    {
        volatile float *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (float *g_idata, float *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

__global__ void reduceForLoop (float *g_idata, float *g_odata, unsigned int n)
{

    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    float *idata = g_idata + blockIdx.x * blockDim.x * 8;

    float *ptr = g_idata + idx;
    float tmp = 0;

    // Increment tmp 8 times with values strided by blockDim.x
    for (int i = 0; i < 8; i++) {
        tmp += *ptr;
        ptr += blockDim.x;
    }
    g_idata[idx] = tmp;

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];

}

int main(int argc, char **argv) {

    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    // A float typically provides up to 7 decimal digits of precision.
    int size = 1 << 16;
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 128;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(float);
    float *h_idata = (float*) malloc(bytes);
    float *h_odata = (float*) malloc(grid.x* sizeof(float));
    float *tmp     = (float*) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (float)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    float gpu_sum = 0.0;

    // allocate device memory
    float *d_idata = NULL;
    float *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(float)));

    // cpu reduction
    iStart = seconds();
    float cpu_sum = recursiveReduce(tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce      elapsed %f sec cpu_sum: %f\n", iElaps, cpu_sum);


    // exercise 3-5: Implement sum reduction of floats in C.
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceForLoop<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0.0;
    for (int i = 0; i < grid.x/8; i++){
        //printf("%d-th gpu_sum BEFORE: %f, h_odata[i]: %f\n", i, gpu_sum, h_odata[i]);
        gpu_sum += h_odata[i];
        //printf("%d-th gpu_sum AFTER:  %f\n", i, gpu_sum);
    }
    printf("gpu reduceUnrollForLoop  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);



    // exercise 3-6: implement version of floats of
    // reduceInterleaved() and reduceCompleteUnrollWarps8().
    // Compare their performance and choose proper
    // metrics and/or events to explain any differences.

    // reduceCompletetUnrollWarps8()
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceCompleteUnrollWarps8<<<grid.x/8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x/8 * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0.0;
    for (int i = 0; i < grid.x/8; i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceCompleteUnrollWarps8  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x/8, block.x);


    // reduceInterleaved()
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduceInterleaved<<<grid.x, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(float),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0.0;
    for (int i = 0; i < grid.x; i++){
        gpu_sum += h_odata[i];
    }
    printf("gpu reduceInterleaved  elapsed %f sec gpu_sum: %f <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // exercise 3-6:
    // reduceInterleaved() and reduceCompleteUnrollWarps8() turned out to be
    // no difference when implemented for floats as integer and float have the
    // same number of bytes. The amount of input and output operations
    // performed doesn't change. Note that float typically provides up to 7
    // decimal digits of precision. e.g. array size of 2**24 could be complicated
    // due to numeric error.

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results

    float diff = (gpu_sum - cpu_sum)*(gpu_sum - cpu_sum);
    bResult = diff < 0.0001; // smaller than epsilon

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
