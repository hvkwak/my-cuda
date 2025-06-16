# Chapter 3 CUDA Execution Model

## 📌 Highlights
- The hardware perspective helps in understanding the nature of kernel execution. Two important features of the CUDA execution model on the GPU are:<br>
    Threads are executed in <strong>warps</strong> in <strong>SIMT</strong> fashion, which is fixed to have <strong>32</strong> threads.
    Hardware resources are partitioned among blocks and threads.
- GPU devices have different compute capabilities. The key of CUDA programming is to optimize the kernel performance under hardware constraints, as introduced in Chapter 2. 
- Optimization of CUDA Execution may involve hiding <em>latency</em> by achieving high <em>occupancy</em> of warps or exposing better parallelism, avoiding <em>branch divergence</em>, <em>unrolling loops</em>, or <em>dynamic paralellism</em> for <em>nested kernel execution</em>.

## 📌 Evaluation Metrics
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
- (`gld_efficiency`) <em>Reqeusted</em> global load throughput is the number of bytes your kernel asked for (e.g., each thread reading 4 bytes).
- (`gld_efficiency`) <em>Required</em> global load throughput is the actual number of bytes the hardware had to transfer from global memory, including inefficiencies due to misalignment, scattered accesses, or uncoalesced loads. ( = `gld_throughput`)
- (`gld_efficiency`) If `gld_efficiency` is 1.0, it means all loads are coalesced, aligned and efficient.
- (`gld_throughput`) Because the minimum memory transaction size is larger than most word sizes, the actual memory throughput required for a kernel can include the transfer of data not used by the kernel. For global memory accesses, this actual throughput is reported by the Global Load Throughput and Global Store Throughput values. (Source: [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/#throughput-reported-by-visual-profiler))
- (`gst_efficiency`) The metric `gst_efficiency`  is the same as `gld_efficiency`, but for global memory stores.
- (`dram_read_throughput`) The metric `dram_write_throughput` is the same as `dram_read_throughput`, but for device memory write throughput.


## 🧪 Exercise 3-1
What are the two primary causes of performance improvement when unrolling loops, data blocks, or warps in CUDA? Explain how each type of unrolling improves instruction throughput.

### 🔑 Answers
1. It reduces instruction overheads.
2. It creates more independent instructions to keep sufficient operations in-flight to saturate instruction and memory bandwidth. Warp scheduler may have more eligible warps that can help hide instruction or memory latency.

## 🧪 Exercise 3-2
Refer to the kernel `reduceUnrolling8()` and implement the kernel `reduceUnrolling16()` in which each thread handles 16 data blocks. Compare kernel performance with `reduceUnrolling8()` and use the proper metrics and events with `nvprof` to explain any difference in performance.


`TODOs below!`
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
