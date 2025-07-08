# Chapter 6 Streams and Concurrency

## 📌 Highlights
- Up to this point, the focus has been on *kernel level concurrency*, in which a single task, or kernel, is executed in parallel by many threads on the GPU, the performance of which depends on the programming model, execution model, and memory model points-of-view.
- In *grid level concurrency*, multiple kernel launches are executed simultaneously on a single device that can lead to better device utilization.
- At the core of this *grid level concurrency*, there are sequences of CUDA operations, namely CUDA *streams*. Streams can be used to implement pipelining, double buffering, or potential concurrent execution of kernels at the granularity of CUDA API calls. The functions in the CUDA API can typically classified as either synchronous(host thread blocking, the `NULL` stream) or asynchronous(returns control to the host immediately after being called, `non-NULL` streams). 
- Non-NULL streams can be further classified into two types: Blocking streams, Non-Blocking streams. (Streams can block other streams, not only the host!)
- Understanding the characteristics of these `NULL` or `non-NULL` streams is the key when it comes to stream synchronization.


## 🧪 Exercise 6-1
Deﬁne the term “CUDA stream.” What kind of operations can be placed in a CUDA stream? What are the main beneﬁts of using streams in an application?

### 🔑 Answers
A CUDA *stream* refers to a sequence of asynchronous CUDA operations that execute on a device in the order issued by the host code. There are mainly two kinds of operations that can be placed in a CUDA stream: memory-related operations and kernel launches. By dispatching kernel execution and memory operations into separate streams, these operations can be overlapped, and the execution time of the prograam can be shortened.

## 🧪 Exercise 6-2
How do events relate to streams? Give an example where a CUDA event would be useful and allow you to implement logic that you could not efﬁciently implement with streams alone.

### 🔑 Answers
`TODO`


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
