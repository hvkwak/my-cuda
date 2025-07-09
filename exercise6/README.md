# Chapter 6 Streams and Concurrency

## 📌 Highlights
- Up to this point, the focus has been on *kernel level concurrency*, in which a single task, or kernel, is executed in parallel by many threads on the GPU, the performance of which depends on the programming model, execution model, and memory model points-of-view.
- In *grid level concurrency*, multiple kernel launches are executed simultaneously on a single device that can lead to better device utilization.
- At the core of this *grid level concurrency*, there are sequences of CUDA operations, namely CUDA *streams*. Streams can be used to implement pipelining, double buffering, or potential concurrent execution of kernels at the granularity of CUDA API calls. The functions in the CUDA API can typically classified as either synchronous(host thread blocking, the `NULL` stream) or asynchronous(returns control to the host immediately after being called, `non-NULL` streams). 
- Non-NULL streams can be further classified into two types: Blocking streams, Non-Blocking streams. (Streams can block other streams, not only the host!)
- Understanding the characteristics of these `NULL` or `non-NULL` streams is the key when it comes to stream synchronization.

## 📌 Characteristics of Streams
- It is firstly important to know whether it's case about the blocking (or non-blocking) with respect to the *host* (thread) OR with respect to *other streams*! Streams can block both of them!
- All operations in non-default streams (`non-NULL`) are non-blocking with respect to the *host*.
- Kernel launches are always, whether `NULL` or `non-NULL` stream, asynchronous with respect to the *host*.
- Memory operations can inherently be synchronous such as `cudaMemcpy`, but asynchronous operation with `cudaMemcpyAsync` is possible. `cudaMemcpy` is not even a stream. it's just synchronous with respect to the host!
- The `NULL` stream, other than `non-NULL` stream, is synchronous stream with respect to the *host*. Most operations added to the `NULL` stream cause the *host* to block on all preceding operations, meaning that the *host* waits for operations that are previously added on the *device*. Kernel launch is exception.
- `Non-NULL` streams can be blocked by the `NULL` stream — if they are created as blocking streams (i.e., without the `cudaStreamNonBlocking` flag).


## 🧪 Exercise 6-1
Deﬁne the term “CUDA stream.” What kind of operations can be placed in a CUDA stream? What are the main beneﬁts of using streams in an application?

### 🔑 Answer
A CUDA *stream* refers to a sequence of asynchronous CUDA operations that execute on a device in the order issued by the *host* code. There are mainly two kinds of operations that can be placed in a CUDA stream: memory-related operations and kernel launches. By dispatching kernel execution and memory operations into separate streams, these operations can be overlapped, and the execution time of the prograam can be shortened.

## 🧪 Exercise 6-2
How do events relate to streams? Give an example where a CUDA event would be useful and allow you to implement logic that you could not efﬁciently implement with streams alone.

### 🔑 Answer
A CUDA *event* is a marker in a CUDA stream associated with a certain point in the flow of operations. A CUDA *event* can let the user monitor device progress and is especially useful when stream executions should be synchronized. For example, the following code snippet shows that `kernelA` runs in `stream1` (non-NULL stream, asynchronous), event is queued in `stream1` after `kernelA`, `stream2` will wait for the `event` (`cudaStreamWaitEvent`), i.e., wait for `kernelA` to finish before running `kernelB`. This kind of synchronization is useful when there are dependencies between multiple streams. It may not be able to efficiently implement this kind of logic with streams alone, as kernel launches are asynchronous with respect to the host thread. 

``` cuda
cudaStream_t stream1, stream2;
cudaEvent_t event;

cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);
cudaEventCreate(&event);

// Launch kernel in stream1
kernelA<<<blocks, threads, 0, stream1>>>();

// Record an event in stream1 after kernelA
cudaEventRecord(event, stream1);

// Tell stream2 to wait for event before continuing
cudaStreamWaitEvent(stream2, event, 0);

// Now launch something in stream2
kernelB<<<blocks, threads, 0, stream2>>>();
```


## 🧪 Exercise 6-3
What can cause false dependencies on GPUs? How do these causes differ between the Fermi and Kepler architectures? How does Hyper-Q help to limit false dependencies?

### 🔑 Answer
In the Fermi GPU, all streams are known to be multiplexed into a single hardware work queue. Its single pipeline may result in a *false dependency*, meaning that only independent task pairs will be executed concurrently. *Hyper-Q* in the Kepler architecture, on the other hand, provides 32 hardware work queues, allowing the GPU to allocate one work queue per stream and thereby enables better concurrency across streams.

## 🧪 Exercise 6-4
Describe the difference between explicit and implicit synchronization. What are examples of speciﬁc CUDA API functions that create implicit host-device synchronization points?

### 🔑 Answer
The difference is that those explicit synchronization functions are inserted manually and are called by the *host* to synchronize the execution of tasks on the *device*. The primary purpose of the functions we often use for implicit synchronization is not actually synchronization itself.
For example, functions like `cudaMemcpy`, `cudaFree` are fundamentally intended for tasks such as data transfer or memory deallocation, but create implicit host-device synchronization points.

## 🧪 Exercise 6-5
`TODO`

### 🔑 Key Ideas
- 



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
