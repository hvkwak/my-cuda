# cuda_programming_works
My CUDA programming exercises codes while reading the book "Professional CUDA C Programming"

## Executing program
Compile and execute the programming exercise of your choice at the project root, e.g. exercise1
```
make EXERCISE=exercise1
./main
```

## Exercise Descriptions
* exercise1: Hello World from GPU. Basic CUDA API calls such as `cudaDeviceReset()`, `cudaDeviceSynchronize()`. Execution based on threadID `threadIdx`. Intro to CUDA online document `CUDA Compiler Driver NVCC`: supported files, gpu-architecture flags, or optimization levels, etc.
