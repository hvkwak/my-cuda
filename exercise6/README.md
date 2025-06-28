# Chapter 6 Streams and Concurrency

## 📌 Highlights
- Up to this point, the focus has been on *kernel level concurrency*, in which a single task, or kernel, is executed in parallel by many threads on the GPU, the performance of which depends on tthe programming model, execution model, and memory model points-of-view.
- In *Grid level concurrency*, multiple kernel launches are executed simultaneously on a single device that often leads to better device utilization.
- Strict ordering in the same CUDA stream, no restriction on kernel execution order in different streams. Streams can be used to implement pipelining or double buffering at the granularity of CUDA API calls. The functions in the CUDA API can typically classified as either synchronous(host thread blocking) or asynchronous(returns control to the host immediately after being called).

