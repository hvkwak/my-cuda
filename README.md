# My CUDA
My CUDA C programming exercises per chapter of the book *Professional CUDA C Programming* by John Cheng, Max Grossman, and Ty McKercher.

## Compilation and Run
Compile and execute the program of your choice at the project root, e.g. exercise1 and `main.cu`
```
make EXERCISE=exercise1 && ./main
```

## Profiling
The command-line profiling tools such as `ncu`(for `CUDA 11.0.194+` ) or `nvprof`(older versions) can help provide detailed insights into kernel performance. For somewhat simpler/customized command-line profiling, run
```
bash ./my_ncu.sh --kernel-name your_kernel ./your_cuda_app
```


## Exercise Descriptions (see per-chapter README.md for details)
* [exercise1](exercise1/): Heterogeneous Parallel Computing with CUDA. Basic CUDA API calls such as `cudaDeviceReset()`, `cudaDeviceSynchronize()`. Execution based on threadID `threadIdx`. Intro to CUDA online document `CUDA Compiler Driver NVCC`: supported files, gpu-architecture flags, or optimization levels, etc.
* [exercise2](exercise2/): CUDA Programming Model. Different execution configuration(varying grid and block size) of `sumArraysOnGPU-timer.cu` to check the difference and reason. Several executions including execution configuration, two elements handling of each thread @ `sumMatrixOnGPU-2D-grid-2D-block.cu`, `sumMatrixOnGPU-2D-grid-1D-block.cu`. Checking maximum size supported by the system for each grid and block dimension @ `checkDeviceInfor.cu`.
* [exercise3](exercise3/): CUDA Execution Model. Introduction to Performance Metrics. General reducing, `Loops Unrolling` and its performance improvements @ `reduceInteger.cu`, `reduceFloat.cu`. Recursive calls in CUDA @ `nestedReduce.cu`.
* [exercise4](exercise4/): Global Memory. Memory transfers using `pageable memory`, `pinned memory`, or `zero copy` @ `memTransfer.cu`, `pinMemTransfer.cu`. Aligned and coalesced memory access patterns @ `readSegment.cu`, `sumArrayZerocpyUVA.cu`. `matrix transpose` example for memory bandwidth check @ `transpose.cu`. performance gain with L1 Cache.
* [exercise5](exercise5/): Shared Memory. Row-major or column-major memory access operations using Shared Memory @ `checkSmemSquare.cu`, Testing `reduceInteger.cu` with different blocksize configuration.
* [exercise6](exercise6/): Streams and Concurrency. Introduction to CUDA Streams. Stream Synchronization. Profiling with `Nsight Systems` timeline
* [exercise8](exercise8/): GPU-Accelerated CUDA Libraries and OpenACC.
