# Chapter 2 CUDA Programming Model

## 📌 Highlights
- Exposing a two-level thread hierarchy through the programming model is one of the CUDA's distinguishing features that lets you control a parallel environment. This two-level thread hierarchy consists of grid of thread blocks and threads in thread blocks. 
- Grid and block dimensions and their configurations could impact on the kernel performance. Learning how to organize threads is one of the central practices of CUDA programming.
- Grids and blocks represent a logical view of the thread layout for kernel execution. Note that naive implementations won't improve the performance.Different perspective, the hardware view, is covered in Chapter 3.

`TODO`: (two-level thread-hierarchy image goes here.)

## 🧪 Exercise 2-1
Using the program `sumArraysOnGPU-timer.cu`, set the `block.x = 1023`. Recompile and run it. Compare the result with the execution configuration of `block.x = 1024`. Try to explain the difference and the reason.

### 🔑 Key Ideas
- Changing the number of blocks leads to different block sizes. If there are more threads in a block, less thread blocks are needed. 
- Less threads requires more thread blocks and this turns out to be the reason for execution time, because only a fixed number of thread blocks can run concurrently that depends on the architecture compute capability.

### 🛠️ Implementation Details
Changing the block size in `main()` to `1023` or `1024`. The grid size will be adjusted accordingly.
``` cuda
// invoke kernel at host side
int iLen = 1023;
dim3 block (iLen);
dim3 grid  ((nElem + block.x - 1) / block.x); // truncated
```

## 🧪 Exercise 2-2
Refer to `sumArraysOnGPU-timer.cu`, and let `block.x = 256.` Make a new kernel to let each thread handle two elements. Compare the results with other execution configurations.

### 🔑 Key Ideas
- That each tread handles two elements can be implemented in various ways. We implement two kernels, where
    `sumArraysOnGPU_cycle()` allows each thread handle two elements in cycling manner and
    `sumArraysOnGPU_neighbor()` allows each thread handle two elements consecutively.

### 🛠️ Implementation Details
`TODO`

### ✅ Execution Results
Having two threads handle two elements that is similar to `Loops Unrolling` turns out to be at least 30% faster than `sumArraysOnGPU();`
`TODO`
```bash
```

## 🧪 Exercise 2-3
Refer to `sumMatrixOnGPU-2D-grid-2D-block.cu`. Adapt it to integer matrix addition. Find the best execution configuration.

### 🔑 Key Ideas
- Several configurations were tested using different values for the 2D dimensions `x` and `y`. A `std::vector` was used to store the set of values `{4, 8, 16, 32, 64}` for each dimension, allowing for a systematic evaluation of various combinations.
- The number of threads per block should be properly tested. The test is skipped, if..
    there are more than 1024 threads per thread block, or
    there are less than 32 threads per thread block.

### 🛠️ Implementation Details
Idea suggested above is pretty self-explanatory. Minimal implementation details can be suggested as below:

``` cuda
// 1. invoke kernel at host side
std::vector<int> dimx_vec = {4, 8, 16, 32, 64};
std::vector<int> dimy_vec = {4, 8, 16, 32, 64};

// 2. skip the configuration
if (dimx * dimy > 1024) {
    printf("Skipping configuration (%d,%d): too many threads per block\n", dimx, dimy);
    continue;
}
if (dimx * dimy < 32){
    printf("Skipping configuration (%d,%d): not enough threads per block\n", dimx, dimy);
    continue;
}
```

### ✅ Execution Results
`TODO`
```bash
```

## 🧪 Exercise 2-4
`TODO`

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```



## 🧪 Exercise 2-5
`TODO`

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```


<!-------------------------------


## 🧪 Exercise 1-2

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```


--------------------------------->
