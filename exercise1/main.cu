#include "../common/common.h"
#include <stdio.h>

void helloFromGPU(dim3 grid, dim3 block);
void helloFromCPU();

int main(void) {

    // hello from CPU
    helloFromCPU();

    // hello from GPU
    dim3 grid(1);
    dim3 block(10);
    helloFromGPU(grid, block);

    // exercise 1-2: see if cudaDeviceReset() makes difference
    // resources e.g. memory won't be released if cudaDeviceReset() were not called here:
    // CHECK(cudaDeviceReset());

    // exercise 1-3: replace cudaDeviceReset() with cudaDeviceSynchronize();
    // without cudaDeviceSynchronize() terminates the host too early, before device prints out the messages.
    CHECK(cudaDeviceSynchronize());

    return 0;
}
