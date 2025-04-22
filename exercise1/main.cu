void helloFromGPU(dim3 grid, dim3 block);
void helloFromCPU();

int main(void) {

    // hello from CPU
    helloFromCPU();

    // hello from GPU
    dim3 grid(1);
    dim3 block(10);
    helloFromGPU(grid, block);
    cudaDeviceReset();
    return 0;
}
