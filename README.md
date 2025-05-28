# cuda_programming_works
My CUDA programming exercises codes from every chapter while reading the book *Professional CUDA C Programming* by John Cheng, Max Grossman, and Ty McKercher.

## Executing program
Compile and execute the programming exercise of your choice at the project root, e.g. exercise1
```
make EXERCISE=exercise1
```

## Exercise Descriptions
* exercise1: Hello World from GPU. Basic CUDA API calls such as `cudaDeviceReset()`, `cudaDeviceSynchronize()`. Execution based on threadID `threadIdx`. Intro to CUDA online document `CUDA Compiler Driver NVCC`: supported files, gpu-architecture flags, or optimization levels, etc.
* exercise2: Introduction to CUDA Programming Model. Different execution configuration(varying grid and block size) of `sumArraysOnGPU-timer.cu` to check the difference and reason. Several executions including execution configuration, two elements handling of each thread @ `sumMatrixOnGPU-2D-grid-2D-block.cu`, `sumMatrixOnGPU-2D-grid-1D-block.cu`. Checking maximum size supported by the system for each grid and block dimension @ `checkDeviceInfor.cu`.
* exercise3: Introduction to CUDA Execution Model. General reducing, `Loops Unrolling` and its performance improvements. Recursive calls in CUDA.
* exercise4: Introduction to Global Memory.
