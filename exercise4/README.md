# Chapter 4 Global Memory

## 📌 Highlights
There are several ways to (dynamically) allocate global memory in the *host* and transfer data to the *device*.
- When transferring the *pageable* host memory, the CUDA driver first allocates temporary *page-locked* or *pinned* host memory, copies the source *pageable* memory to it, and then transfers the data from pinned memory to *device* memory.
- *Pinned* memory is host memory that is locked in physical RAM. Its data transfer between *host* and *device* is faster than *pageable* memory. For example, on Fermi devices, it is advantageous to use *pinned* memory when transferring more than 10MB of data. Note that Too much *pinned* memory leads to less *pageable* memory on host. It can degrade system performance and has slow allocation time.
- *Zero-copy* memory is a type of pinned (non-pageable) memory that is mapped into the device address space. It enables the GPU to directly access the host memory without needing to transfer it to the device memory.
- With *Unified Virtual Addressing*, *host* and *device* memory share a single virtual address space.
- *Unified Memory* is a new feature that simplifies memory management in the CUDA programming model.
- Data transfer from *host* to *device* is expensive and may be a reason for a bottleneck of the overall application performance, if not managed properly. Thereotical peak bandwidth between GPU chip and the on-board GPU memory is very high. The link between CPU and GPU through the PCIe Bus is limited which varies by the architecture.

`TODO`: add here the picture of pageable/pinned memory!

## 🧪 Exercise 4-2
Refer to the file `globalVariable.cu`. Replace the following symbol copy functions:
`cudaMemcpyToSymbol()`
`cudaMemcpyFromSymbol()`

with the data transfer function

`cudaMemcpy()`

You will need to acquire the address of the global variable using
`cudaGetSymbolAddress()`

### 🔑 Key Ideas
- Basic but key: it's about static global memory!
- `cudaMemcpyToSymbol()` copies data from *host* to global memory in *device*,
- `cudaMemcpyFromSymbol()` copies data from global memory in *device* to *host*. 
- `cudaGetSymbolAddress()` allows direct mapping of an address of a symbol in *device*
- `cudaMemcpy()` can directly copy pageable *host* memory to global memory that a pointer is pointing to.

### 🛠️ Implementation Details
Resetting `valueArray` is added to test it properly.

``` cuda
// (..snipped..)
float valueArray[5] = {3.14f, 3.14f, 3.14f, 3.14f, 3.14f};
CHECK(cudaMemcpyToSymbol(devArray, &valueArray, sizeof(float)*5)); // To symbol "devArray"
for(int i = 0; i < 5; i++){
    printf("Host:   copied %f to the global array \n", valueArray[i]);
}

checkGlobalArray<<<1, 5>>>();

CHECK(cudaMemcpyFromSymbol(&valueArray, devArray, sizeof(float)*5)); // From symbol "devArray"
for(int i = 0; i < 5; i++){
    printf("Host:   the value changed by the kernel to %f\n", valueArray[i]);
    valueArray[i] = 3.14f; // reset
}

// (..snipped..)
float *dptr = NULL; // pointer to the first element of devArray
CHECK(cudaGetSymbolAddress((void **)&dptr, devArray));
CHECK(cudaMemcpy(dptr, &valueArray, sizeof(float)*5, cudaMemcpyHostToDevice));

// (..snipped..)
```

## 🧪 Exercise 4-4
Compare performance of the pageable and pinned memory copies in `memTransfer.cu` and `pinMemTransfer.cu` using CPU Timers and different sizes: 2M, 4M, 8M, 16M, 32M, 64M, 128M.

### 🔑 Key Ideas
- `memTransfer.cu` uses the pageable memory copies. This involves ...<br>
    `malloc()` that allocates the (pageable) *host* memory and<br>
    `cudaMalloc()` that allocates the *device* memory,<br>
- `pipnMemTransfer.cu` uses pinned memory copies. This involves ...<br>
    `cudaMallocHost()` that allocates the pinned *host* memory and<br>
    `cudaMalloc()` that allocates the *device* memory. // analog in `memTransfer.cu`<br>

where `cudaMemcpy()` that transfers data between the *host* and *device* memory is used for both cases. `cudaFree()` deallocates the memory in *device*. Note that `cudaHostAlloc()` with the flag `cudaHostAllocDefault` may replace `cudaMallocHost()`.


### ✅ Execution Results
Pinned memory transfer shows significantly faster data transfer times compared to standard (pageable) host memory, but incurs a higher allocation overhead.

GPU Standard Memory Transfer (Pageable host memory)
| Memory Size | Transfer Time (s) | Alloc Time (s) | Dealloc Time (s) |
|-------------|-------------------|----------------|------------------|
| 2.00 MB     | 0.000972          | 0.000015       | 0.000143         |
| 4.00 MB     | 0.001254          | 0.000017       | 0.000107         |
| 8.00 MB     | 0.002268          | 0.000017       | 0.000096         |
| 16.00 MB    | 0.004221          | 0.000016       | 0.000109         |
| 32.00 MB    | 0.008255          | 0.000017       | 0.000183         |
| 64.00 MB    | 0.016605          | 0.000023       | 0.000280         |
| 128.00 MB   | 0.032237          | 0.000019       | 0.000341         |

GPU Pinned Memory Transfer
| Memory Size | Transfer Time (s) | Alloc Time (s) | Dealloc Time (s) |
|-------------|-------------------|----------------|------------------|
| 2.00 MB     | 0.000487          | 0.001930       | 0.000137         |
| 4.00 MB     | 0.000727          | 0.003317       | 0.000124         |
| 8.00 MB     | 0.001403          | 0.006086       | 0.000125         |
| 16.00 MB    | 0.002795          | 0.012979       | 0.000143         |
| 32.00 MB    | 0.005414          | 0.018640       | 0.000123         |
| 64.00 MB    | 0.011070          | 0.036481       | 0.000280         |
| 128.00 MB   | 0.022400          | 0.073413       | 0.000333         |

## 🧪 Exercise 4-5
Modify `sumArrayZerocopy.cu` to access A, B, and C at an offset. Compare performance with and without L1 cache enabled. If your GPU does not support conﬁguring the L1 cache, reason about the expected results.

### 🔑 Key Ideas
- If the memory access patterns are aligned and consecutive, accesses are coalesced into a single transaction (of a warp).
- Adding an offset can prevent the threads from coalesced access.

### 🛠️ Implementation Details
Adding the `offset` parameter will modify the memory access pattern.
``` cuda
void sumArraysOnHost(float *A, float *B, float *C, const int N, int offset)
{
    // exercise 4-5: Modify access at an offset.
    for (int idx = offset, k = 0; idx < N; idx++, k++)
    {
        C[k] = A[idx] + B[idx]; // k: 0 ~ (N-offset)
    }
}

__global__ void sumArrays(float *A, float *B, float *C, const int N, int offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int k = i + offset;

    if (k < N) C[i] = A[k] + B[k];
}
```

### 📊 Kernel Performance Comparison: `offset = 0` vs `offset = 2`
Offset causes misalignment and wasted cache lines, which is reflected in the drop in `gld_efficiency`. To compensate for the inefficiency, the global load throughput (`gld_throughput`) increases when `offset = 2`, resulting in higher memory traffic. Despite the higher throughput, the overall execution time is slightly longer compared to the aligned case (`offset = 0`). 

Configuring L1 cache is not supported on the GTX 1660 (Turing architecture), and in this case, L1 cache likely has minimal impact on performance. Since each memory location is accessed only once, most cache lines are either used once or partially wasted, reducing the benefits of caching.

| Metric                   | `offset = 0`    | `offset = 2`    |
|--------------------------|-----------------|-----------------|
| **inst_per_warp**        | 16 inst/warp    | 16 inst/warp    |
| **inst_executed**        | 2,097,152       | 2,097,152       |
| **inst_per_cycle**       | 0.10 inst/cycle | 0.10 inst/cycle |
| **achieved occupancy**   | 82.64%          | 81.97%          |
| **sm_efficiency**        | 94.14%          | 94.49%          |
| **gld_throughput**       | 198.22 GB/s     | **244.58 GB/s** |
| **gld_efficiency**       | **100%**        | **80.00%**      |
| **gst_throughput**       | 99.11 GB/s      | 97.83 GB/s      |
| **gst_efficiency**       | 100%            | 100%            |
| **dram_read_throughput** | 198.35 GB/s     | 195.80 GB/s     |
| **duration**             | **169.28 µs**   | **171.49 µs**   |

### Note
This can be further examined in `readSegment.cu` for exercise 4-7, 4-8, and 4-9.


## 🧪 Exercise 4-15
Refer to the kernel `tranposeUnroll4Row`. Implement a new kernel, `tranposeRow`, to let each thread handle all elements in a row. Compare the performance with existing kernels and use proper metrics to explain the difference.

### 🔑 Key Ideas
- Let each thread handle all elements in a row. This leads to smaller size of grid and block.
- Note that this shows performance loss due to poor strided memory access patterns. Read addresses of threads in a warp should be consecutive (coalesced).
- Stores, on the other hand, are coalesced, resulting in high store efficiency. However, the measured store throughput may still appear low, since the kernel spends a significant amount of time stalled waiting for uncoalesced loads to complete. The stores execute quickly, but not often enough to raise the average throughput.


### 🛠️ Implementation Details
One for-loop is added to let thread handle all elements in a row. 
``` cuda
__global__ void transposeRow(float *out, float *in, const int nx, const int ny){

    // unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x; // 0
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;

    unsigned int ti; // unrolled access in rows
    unsigned int to;

    // thread handles all elements in a row.
    for (int ix = 0; ix < nx; ix++){

        ti = iy * nx + ix;
        to = ix * ny + iy;

        if (iy < ny){
            out[to] = in[ti];
        }
    }
}
```

### 📊 Kernel Performance Comparison: `transposeNaiveRow` vs `transposeRow`
A comparison with the original implementation, `transposeNaiveRow`, highlights the key differences clearly. In `transposeNaiveRow`, each thread handles one matrix element, while in `transposeRow`, each thread processes an entire row. This difference leads to significant variation in metrics like `inst_per_warp`, `inst_executed`, and `inst_per_cycle`.

`transposeRow` launches far fewer threads, which results in poor warp scheduling, extremely low achieved occupancy and lower `sm_efficiency`. Despite this, it achieves 100% store efficiency, as its memory access pattern for writes is fully coalesced.

However, `transposeRow` is over 2× slower than `transposeNaiveRow` in GTX 1660. This is primarily due to uncoalesced global load instructions, which cause frequent stalls. These stalls also explain the relatively low `gst_throughput`, despite perfect store efficiency — the stores are delayed by the earlier load instructions.

| Metric                   | `transposeNaiveRow` | `transposeRow`   |
|--------------------------|---------------------|------------------|
| **inst_per_warp**        | 18 inst/warp        | 21,528 inst/warp |
| **inst_executed**        | 2,359,296           | 1,377,792        |
| **inst_per_cycle**       | 0.08 inst/cycle     | 0.03 inst/cycle  |
| **achieved occupancy**   | 81.31%              | 9.10%            |
| **sm_efficiency**        | 99.45%              | 70.25%           |
| **gld_throughput**       | 74.29 GB/s          | 271.25 GB/s      |
| **gld_efficiency**       | 100%                | 12.50%           |
| **gst_throughput**       | 297.17 GB/s         | 33.91 GB/s       |
| **gst_efficiency**       | 25%                 | 100%             |
| **dram_read_throughput** | 74.46 GB/s          | 34.04 GB/s       |
| **duration**             | 225.82 µs           | 494.82 µs        |





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
