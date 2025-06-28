# Chapter 4 Global Memory

## 📌 Highlights
- There are several ways to allocate global memory in the *host* and transfer data to the *device*.
- When transferring the *pageable* host memory, the CUDA driver first allocates temporary *page-locked* or *pinned* host memory, copies the source *pageable* memory to it, and then transfers the data from pinned memory to *device* memory.
- *Pinned* memory is host memory that is locked in physical RAM. Its data transfer between *host* and *device* is faster than *pageable* memory. For example, on Fermi devices, it is advantageous to use *pinned* memory when transferring more than 10MB of data. Note that Too much *pinned* memory leads to less *pageable* memory on host. It can degrade system performance and has slow allocation time.
- *Zero-copy* memory is a type of pinned (non-pageable) memory that is mapped into the device address space. It enables the GPU to directly access the host memory without needing to transfer it to the device memory.
- With *Unified Virtual Addressing*, *host* and *device* memory share a single virtual address space.
- *Unified Memory* is a new feature that simplifies memory management in the CUDA programming model.
- Data transfer from *host* to *device* is expensive and may be a reason for a bottleneck of the overall application performance, if not managed properly. Thereotical peak bandwidth between GPU chip and the on-board GPU memory is very high. The link between CPU and GPU through the PCIe Bus is limited which varies by the architecture.



## 🧪 Exercise 4-1
Refer to the ﬁle `globalVariable.cu`. Declare statically a global ﬂoat array with a size of ﬁve elements. Initialize the global array with the same value of 3.14. Modify the kernel to let each thread change the value of the array element with the same index as the thread index. Let the value be multiplied with the thread index. Invoke the kernel with ﬁve threads.


### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```



## 🧪 Exercise 4-2
Refer to the file `globalVariable.cu`. Replace the following symbol copy functions:
`cudaMemcpyToSymbol()`
`cudaMemcpyFromSymbol()`

with the data transfer function

`cudaMemcpy()`

You will need to acquire the address of the global variable using
`cudaGetSymbolAddress()`

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```



## 🧪 Exercise 4-3
Compare performance of the pinned and pageable memory copies in `memTransfer` and `pinMemTransfer` using `nvprof` and different sizes: 2M, 4M, 8M, 16M, 32M, 64M, 128M.

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```

## 🧪 Exercise 4-4
Using the same examples, compare the performance of pinned and pageable memory allocation and deallocations using CPU timers and different sizes: 2M, 4M, 8M, 16M, 32M, 64M, 128M.

### 🔑 Key Ideas
- 

### 🛠️ (Optional) Implementation Details

### 📈 (Optioinal) Performance Metrics

### ✅ Execution Results
```bash
```

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
