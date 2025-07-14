## Chapter 5 Shared Memory (and Constant Memory)

## 📌 Highlights
- Shared memory is a program-managed cache within the CUDA memory hierarchy that can be explicitly controlled to significantly improve performance.
- There are two main reasons to use shared memory in a CUDA kernel. One is to cache data on-chip, which helps reduce the number of global memory transactions. The other is to rearrange data in shared memory in a way that avoids non-coalesced global memory accesses, leading to more efficient memory access.
- The shared memory address space is partitioned among all resident thread blocks and is shared by all threads within a thread block. Its contents have the same lifetime as the thread block in which it was created and may limit kernel occupancy due to limited size of shared memory.
- Shared memory acesses are issued per warp. Ideally, each request to access shared memory by a warp is serviced in one transaction. To achieve this, or high memory bandwidth, note that shared memory is divided into 32 equally-sized memory modules *banks*, which can be accessed simultaneously. When multiple separate address request in a shared memory fall into the same memory bank, a *bank conflict* occurs, causing the transaction to be replayed. The art of CUDA programming that implements shared memory access is subject to avoid this *bank conflict*, where *padding* would be a reasonable option.
- Constant memory is optimized for read-only data that is broadcast to many threads at a time.
  <div style="display: inline-block; vertical-align: top;">
    <img src="images/Figure5-1.png" alt="Figure 5-1. The CUDA Memory Hierarchy (Cheng et al.)" width="500"><br>
    <strong>Figure 5-1. The CUDA Memory Hierarchy. There is shared memory per SM.  (Cheng et al.)</strong><br>
  </div>

## 🧪 Exercise 5-1
Suppose you have a shared memory tile with dimensions [32][32]. Pad a column to it and then draw an illustration showing the mapping between data elements and banks for a Kepler device in 4-byte access mode.

### 🔑 Illustration
<div style="display: inline-block; vertical-align: top;">
  <img src="images/Padding.png" alt="Padding an additional column to a shared memory tile with dimensions 32x32" width="500"><br>
</div>
- Shared memory acesses are issued per warp of 32 threads. 
- Data elements are divided into 32 equally-sized memory modules *banks*. Each box's index indicates the allocated memory banks.
- The red-highlighted column represents padding — an extra column added to each row in shared memory. This avoids bank conflicts of column-major access by offsetting each row’s starting address, ensuring that accesses fall into different banks instead of all hitting the same one.
- In row-major access, thread `t` in a warp accesses element `[row][t]`. As `t` ranges from `0` to `31`, each thread accesses a different bank `(t % 32)`, leading to parallel and conflict-free access.<br>


### 📈 Note: No profile available for Exercise 5-2, 5-3, and 5-4.
`nvprof` metrics `shared_load_transactions_per_request` and `shared_store_transactions_per_request` are no longer available on the `PerfWorks Metric or Formula (>= SM 7.0)` list! Sorry!

## 🧪 Exercise 5-2
Refer to the kernel `setRowReadCol` in the ﬁle `checkSmemSquare.cu`. Make a new kernel named `setColReadRow` to perform the operations of writing to columns and reading from rows. Check the memory transactions with `nvprof` and observe the output.

### 🔑 Key Ideas
- Use `threadIdx.x` for column index and `threadIdx.y` for row index for store.

### 🛠️ Implementation Details
Easy!
``` cuda
__global__ void setColReadRow(int *out)
{
    //static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x];
}
```

### ✅ Execution Results
A shared memory block with the size of (4, 4) was tested. `shared_load_transactions_per_request` is expected to be 1, whereas `shared_store_transactions_per_request` is expected to be greater than 1, though may vary by architecture.
``` bash
root@ubuntu:/workspace/cuda_programming_works# ./checkSmemSquare
./checkSmemSquare at device 0: NVIDIA GeForce RTX 4090 with Bank Mode:4-Byte <<< grid (1,1) block (4,4)>>>
set row read col   :     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set col read row   :     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
```

## 🧪 Exercise 5-3
Refer to the kernel `setRowReadColDyn` in the ﬁle `checkSmemSquare.cu`. Make a new kernel named `setColReadRowDyn` that declares shared memory dynamically, and then perform the operations of writing to columns and reading from rows. Check the memory transactions with `nvprof` and observe the output.

### 🔑 Key Ideas
- Dynamic shared memory must be declared as an unsized 1D array; we need to calculate memory access indices based on 2D thread indices.
- `row_idx`: 1D row-major memory offset calculated from 2D thread indices.
- `col_idx`: 1D column-major memory offset calculated from 2D thread indices

### 🛠️ Implementation Details
``` cuda
    // snipped
    unsigned int row_idx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int col_idx = threadIdx.x * blockDim.y + threadIdx.y;
```

### ✅ Execution Results
The store operation is expected to have bank conflict, whereas the load operations are coalesced. No profiling!
```bash
root@ubuntu:/workspace/cuda_programming_works# ./checkSmemSquare
./checkSmemSquare at device 0: NVIDIA GeForce RTX 4090 with Bank Mode:4-Byte <<< grid (1,1) block (4,4)>>>
set row read col dyn:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set col read row dyn:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
```

## 🧪 Exercise 5-4
Refer to the kernel `setRowReadColPad` in the ﬁle `checkSmemSquare.cu`. Make a new kernel named `setColReadRowPad` that pads by one column. Then, implement the operation of writing by columns and reading from rows. Check the memory transactions with `nvprof` and observe the output.

### 🔑 Key Ideas
- Adding one more column for padding will may prevent the store operation from *bank conflict*.
- Note that the padded column is not used to store data.

### 🛠️ (Optional) Implementation Details
``` cuda
    // snipped
    __shared__ int tile[BDIMY][BDIMX + IPAD]; // static shared memory
```

### ✅ Execution Results
The store operation is expected be free of bank conflict.
```bash
root@ubuntu:/workspace/cuda_programming_works# ./checkSmemSquare
./checkSmemSquare at device 0: NVIDIA GeForce RTX 4090 with Bank Mode:4-Byte <<< grid (1,1) block (4,4)>>>
set row read col pad:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
set col read row pad:     0    4    8   12    1    5    9   13    2    6   10   14    3    7   11   15
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
