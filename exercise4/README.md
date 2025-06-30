# Chapter 4 Global Memory

## 📌 Highlights
There are several ways to (dynamically) allocate global memory in the *host* and transfer data to the *device*.
- When transferring the *pageable* host memory, the CUDA driver first allocates temporary *page-locked* or *pinned* host memory, copies the source *pageable* memory to it, and then transfers the data from pinned memory to *device* memory.
- *Pinned* memory is host memory that is locked in physical RAM. Its data transfer between *host* and *device* is faster than *pageable* memory. For example, on Fermi devices, it is advantageous to use *pinned* memory when transferring more than 10MB of data. Note that Too much *pinned* memory leads to less *pageable* memory on host. It can degrade system performance and has slow allocation time.
- *Zero-copy* memory is a type of pinned (non-pageable) memory that is mapped into the device address space. It enables the GPU to directly access the host memory without needing to transfer it to the device memory.
- With *Unified Virtual Addressing*, *host* and *device* memory share a single virtual address space.
- *Unified Memory* is a new feature that simplifies memory management in the CUDA programming model.
- Data transfer from *host* to *device* is expensive and may be a reason for a bottleneck of the overall application performance, if not managed properly. Thereotical peak bandwidth between GPU chip and the on-board GPU memory is very high. The link between CPU and GPU through the PCIe Bus is limited which varies by the architecture.


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
    `malloc()` that allocates the (pageable) *host* memory,<br>
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
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```

## 🧪 Exercise 1-2

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
