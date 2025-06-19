# Chapter 3 CUDA Execution Model

## 📌 Highlights
- The hardware perspective helps in understanding the nature of kernel execution. Two important features of the CUDA execution model on the GPU are:<br>
    Threads are executed in <strong>warps</strong> in <strong>SIMT</strong> fashion, which is fixed to have <strong>32</strong> threads.
    Hardware resources are partitioned among blocks and threads.
- GPU devices have different compute capabilities. The key of CUDA programming is to optimize the kernel performance under hardware constraints, as introduced in Chapter 2. 
- Optimization of CUDA Execution may involve hiding <em>latency</em> by achieving high <em>occupancy</em> of warps or exposing better parallelism, avoiding <em>branch divergence</em>, <em>unrolling loops</em>, or <em>dynamic paralellism</em> for <em>nested kernel execution</em>.

## 📈 Performance Metrics - Profiling Tools
The command-line profiling tools such as `ncu`(for `CUDA 11.0.194+` ) or `nvprof`(older versions) can help provide detailed insights into kernel performance. Some of the important performance evaluation metrics that are mentioned in this book are introduced below:

| Metric               | Description                                                                                                                     |
|----------------------|---------------------------------------------------------------------------------------------------------------------------------|
| branch_efficiency    | Ratio of non-divergent branches to total branches. Branch Efficiency = 100 * (#Branches - #Divergent Branches)/#Branches        |
| achieved_occupancy   | Ratio of the average active warps per cycle to the maximum number of warps supported on an SM. (optional: per device occupancy) |
| gld_efficiency       | Ratio of <em>requested</em> global load throughput to <em>required</em> global load throughput.                                 |
| gld_throughput       | The amount of data actually required for a kernel (e.g. per second)                                                             |
| gst_efficiency       | Ratio of <em>requested</em> global store throughput to <em>required</em> global store throughput.                               |
| inst_per_warp        | The Average number of instructions executed by each warp.                                                                       |
| dram_read_throughput | Device memory read throughput. (Available for compute capability 5.0 and 5.2.)                                                  |
| stall_sync           | Percentage of stalls occurring because the warp is blocked at a `__syncthreads()` call                                          |


Notes
- (gld_efficiency) <em>Reqeusted</em> global load throughput is <em>the number of bytes your kernel asked for</em> (e.g., each thread reading 4 bytes).
- (gld_efficiency) <em>Required</em> global load throughput is <em>the actual number of bytes the hardware had to transfer</em> from global memory, including inefficiencies due to misalignment, scattered accesses, or uncoalesced loads. ( = gld_throughput)
- (gld_efficiency) If gld_efficiency is 1.0, it means all loads are coalesced, aligned and efficient.
- (gld_throughput) Because the minimum memory transaction size is larger than most word sizes, the actual memory throughput required for a kernel can include the transfer of data not used by the kernel. For global memory accesses, this actual throughput is reported by the Global Load Throughput and Global Store Throughput values. (Source: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-reported-by-visual-profiler))
- (gst_efficiency) The metric gst_efficiency is the same as gld_efficiency, but for global memory stores.
- (dram_read_throughput) The metric dram_write_throughput is the same as dram_read_throughput, but for device memory write throughput.
where it is to note that the term <em>throughput</em>, however, is the amount of operations that can be processed per unit of time, commonly expressed as gflops (which stands for billion floating-point operations per second) (or bytes/sec could be the case when it comes to `load` operations), especially in fields of scientific computation that make heavy use of floating-point calculations. 


## 🧪 Exercise 3-1
What are the two primary causes of performance improvement when unrolling loops, data blocks, or warps in CUDA? Explain how each type of unrolling improves instruction throughput.

### 🔑 Answers
1. It reduces instruction overhead that is associated with each iteration.
2. It creates more independent instructions to keep sufficient operations in-flight to saturate instruction (instruction-level parallelism) and memory bandwidth. Additionally, with more work per thread, warp schedulers are more likely to have more eligible warps ready to execute that can help hide instruction or memory latency.

## 🧪 Exercise 3-2
Refer to the kernel `reduceUnrolling8()` and implement the kernel `reduceUnrolling16()` in which each thread handles 16 data blocks. Compare kernel performance with `reduceUnrolling8()` and use the proper metrics and events with `nvprof` to explain any difference in performance.

### 🔑 Key Ideas
- Hard-coded unrolling of 16 Elements. 
- In-place reduction takes place based on the stride, which will be reduced by half per loop.
- In-place reduction is implemented with a for-loop. If the number of elements is known, this could be further hard-coded, or `#pragma unroll` directive could be an option. 
- The number of thread blocks is reduced by a factor of 16 accordingly.

### 🛠️ Implementation Details
``` cuda
__global__ void reduceUnrolling16 (int *g_idata, int *g_odata, unsigned int n){
    
    // unrolling 16
    if (idx + 15 * blockDim.x < n){
        int a1 = g_idata[idx];
        int a2 = g_idata[idx + blockDim.x];
        int a3 = g_idata[idx + 2 * blockDim.x];
        int a4 = g_idata[idx + 3 * blockDim.x];
        int b1 = g_idata[idx + 4 * blockDim.x];
        // (... total 16 lines of hard-coding snipped ...)
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4 + c1 + c2 + c3 + c4 + d1 + d2 + d3 + d4;
    }
    
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

// (... snipped ...)
reduceUnrolling16<<<grid.x / 16, block>>>(d_idata, d_odata, size);
```

### ✅ Execution Results
`TODO`
```bash
```

## 🧪 Exercise 3-3
Refer to the kernel reduceUnrolling8 and replace the following code segment:
```cuda
int a1 = g_idata[idx];
int a2 = g_idata[idx+blockDim.x];
int a3 = g_idata[idx+2*blockDim.x];
int a4 = g_idata[idx+3*blockDim.x];
int b1 = g_idata[idx+4*blockDim.x];
int b2 = g_idata[idx+5*blockDim.x];
int b3 = g_idata[idx+6*blockDim.x];
int b4 = g_idata[idx+7*blockDim.x];
g_idata[idx] = a1+a2+a3+a4+b1+b2+b3+b4;
```

with the functionally equivalent code below:

``` cuda
int *ptr = g_idata + idx;
int tmp = 0;
// Increment tmp 8 times with values strided by blockDim.x
for (int i = 0; i < 8; i++) {
    tmp += *ptr; ptr += blockDim.x;
}
g_idata[idx] = tmp;
```
Compare the performance of each and explain the difference using `nvprof` metrics.

### ✅ Execution Results
`TODO`
```bash
```

## 🧪 Exercise 3-4
Refer to the kernel `reduceCompleteUnrollWarps8`. Instead of declaring vmem as `volatile`, use `__syncthreads`. Note that `__syncthreads` must be called by all threads in a block. Compare the performance of the two kernels. Use nvprof to explain any differences.

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```


## 🧪 Exercise 3-5
Implement sum reduction of floats in C.

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```

## 🧪 Exercise 3-6
Refer to the kernel `reduceInterleaved` and the kernel `reduceCompleteUnrollWarps8` and implement a version of each for floats. Compare their performance and choose proper metrics and/or events to explain any differences. Are there any differences compared to operating on integer data types?

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```
## 🧪 Exercise 3-7
When are the changes to global data made by a dynamically spawned child guaranteed to be visible to its parent?

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```


## 🧪 Exercise 3-8
Refer to the ﬁle `nestedHelloWorld.cu` and implement a new kernel using the methods illustrated in Figure 3-30.

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```

## 🧪 Exercise 3-9
Refer to the ﬁle `nestedHelloWorld.cu` and implement a new kernel that can limit nesting levels to a given depth.

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
