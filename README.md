CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jacqueline (Jackie) Li
  * [LinkedIn](https://www.linkedin.com/in/jackie-lii/), [personal website](https://sites.google.com/seas.upenn.edu/jacquelineli/home), [Instagram](https://www.instagram.com/sagescherrytree/), etc.
* Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)

# Program Overview

I do not know what I am doing send help pls.

This is a stream compaction program which has a CPU, Naive, and Efficient implementation of the stream compaction algorithm. 

Extra credit implemented: Radix Sort.

#### CPU Implementation

Idea: For the scan algorithm, launch one thread to linearly calculate a prefix sum. Then, in the stream compaction portion, use a flag to indicate which items are to be included in the final array, run scan, then add the data corresponding to the scanned array indices to the output array. 

For this version of stream compaction, the include condition is if element of array is not 0.

Pseudocode:

```
# Scan.
Set first element of out array to 0.

for i from 1 -> size of array do:
  # Accumulate out with last value of in, exclusive scan.
  out[i] = out[i - 1] + in[i - 1]

# Compact without scan.

Initialize counter.
for i from 0 -> n do:
  if in[i] != 0 (condition) :
    Set out array's data at the counter index to in[i]
    The reason to do this is b/c counter keeps track of when the element passes the filtering condition. 
    counter++

# Compact with scan.
Calculate flag array: 1 for item to be included, 0 o.w.

Call scan function on flag array.

Scatter step:
Initialize counter. 
for i from 0 -> n do:
  if flag[i] == 1 (means elem included):
    Set out array's data at scanned[i] index to in[i].
    Because we run scan on the flag array, scanned array will hold the new indices of filtered array with given condition.
    counter++
```

Runtime: O(n)

#### Naive Implementation

Idea: Divide each summation into parallel processes which propagate each sum of two numbers down to the next process. Each summation is now a partial sum which will calculate a sum with its 2^(d-1)th neighbour. 

```
Create double buffers to ping pong values as parallel sum is computed by layer

for d from 1 -> log_2(n) do:
  Get offset of nearest neighbour to second power
  offset = 1 << (d - 1)
  if current thread index > offset (in neighbouring thread):
    out[index] = in[index] + in[index - offset]
    Take partial sum of current point and neighbour thread point to obtain out value
  else:
    out[index] = in[index]
```

Runtime: O(nlog_2(n))

#### Efficient Implementation

Idea: Use a balanced binary tree to conduct scanning, implementing an upsweep (parallel reduction) then a downsweep to build the scan in place using partial sums. The downsweep takes the n/2 sum to the left and the n sum on the right and accumulates through ceil(log_2(n)) levels. Integrate that into stream compaction by using the same three step process described in naive, but run on kernels instead (flag, scan, scatter). 

```
Initialize single buffer

# Upsweep.
Parallel reduction.
for d from 0 -> ceil(log_2(n)) do:
  Multiply k by 1 << (d + 1)
  slideR = 1 << (d + 1)
  slideL = 1 << (d)
  Sum in[k + slideR - 1] with in[k + slideL]

# Downsweep.
for d from ceil(log_2(n)) -> 0 do:
  Multiply k by 1 << (d + 1)
  Each parent node propagates sum down to left child
  Each right child inherits left child's old value + parent node value

# Stream compaction with efficient scan.
Use kernel function to map all elements according to provided condition.

Call scan function to obtain a scanned array.

Call scatter to accumulate scanned and flagged arrays, and compute final output array.
```

Runtime: O(n)

#### Thrust Implementation

This implementation just calls the exclusive scan algorithm from the thrust API, which we mainly do to compare the runtime with the other methods.

#### Extra Credit: Radix Sort

Idea: Keep partitioning data by bits, from least significant bit to most significant bit. On each pass, divide element by two groups - 0 on the left and 1 on the right. Use the process for stream compaction (flag, scan, scatter) to rearrange the elements. By the end, you should end up with a sorted array based on the bits. 

```
For each tile of number in an array:

Create buffers for array passes.
- input, output
- b: boolean flag array for current bit
- e: inverse of b
- f: exclusive scan result on e
- t: addresses for values that pass condition

for bit from 0 -> (sizeof(int)*8 - 1): 

  # Compute inverse bits
  For each element in parallel:
    b[i] = (idata[i] >> bit) & 1
    e[i] = 1 - b[i]

  # Scan
  Run exclusive scan on f

  # Calculate total false
  totalFalse = total numnber of zeroes

  For each element in parallel, i is index:
    t[i] = i - f[i] + totalFalse 

  # Use boolean flag array and scanned array to update output
  For each element in parallel, i is index:
    if b[i] == 0:
       odata[f[i]] = idata[i]
    else:
       odata[t[i]] = idata[i]

  Swap(idata, odata)
```

Runtime: O(n) or O(n * 32)

## Runtime Analysis

For the first set of runtime analysis tests, I varied the size of the array and plotted the runtime of each method against each other. I hypothesise that the CPU version will take exponentially more time than the naive and efficient methods because the CPU version only launches one thread at a time to calculate, while at least naive and efficient algorithm engage in parallel processing.

### Size of Array v. Runtime of Scan Method

**blockSize = 128, array size power of 2**

| Array Size     | CPU  | Naive | Efficient |
|----------------|------|-------|-----------|
| 21             |3.424 | 1.434 |   1.735   |
| 22 (default)   |8.046 | 2.875 |   3.905   |
| 23             |14.966| 5.634 |   3.861   |
| 24             |33.699| 11.739|   6.451   |

| ![](images/arrSize_scanPow2.png) |
|:--:|

**blockSize = 128, array size non power of 2**

| Array Size     | CPU  | Naive | Efficient |
|----------------|------|-------|-----------|
| 21             |3.223 | 1.304 |   1.923   |
| 22 (default)   |8.760 | 2.799 |   3.272   |
| 23             |15.419| 5.486 |   3.585   |
| 24             |38.369| 11.443|   7.157   |

| ![](images/arrSize_scanNonPow2.png) |
|:--:|

From these two graphs, we can clearly see that the timing for the CPU methodology increases exponentially as the array size increases, which is to be expected because for larger data values CPU memory will spill over L1/L2 cache, making the sequential loop take more time to retrieve values. 

In the naive GPU implementation, a kernel is launched every loop for log(n) times, where n is the length of the array. We can see compared to efficient, naive actually beats it for smaller array sizes, because it requires fewer threads. Naive will launch in total log(n) kernels, and its memory is less coalesced because as offset increases, it will take longer to access memory that is further away (i.e. neighbour is a lot further, so takes more time to access neighbour memory even if the computation still takes minimal time).

In the efficient implementation, we launch a kernel each for upsweep and downsweep, for d from 0 -> log_ceil2(n) times. So in total, 2 * log(n) kernels are launched. The reason why efficient is faster than naive though even though it launches more kernels is because naive has several issues, including but not limited to idle threads as the offset increases, reading and writing overlapping values, and that memory is less coalesced because every thread reads from an increasingly distant neighbour. Hence, efficient implementation runtime will ultimately trump both CPU and naive for larger array sizes, because upsweep and downsweep only touch each element log(n) times, leading to less idle threads and memory that is more coalesced, meaning threads can access consecutive memory threads, hence making runtime faster.

### Size of Array v. Runtime of Stream Compaction Method

**blockSize = 128, array size power of 2**

| Array Size     | CPU   | Efficient |
|----------------|-------|-----------|
| 21             |10.845 |   2.366   |
| 22 (default)   |23.797 |   3.163   |
| 23             |58.646 |   5.707   |
| 24             |129.814|   10.458  |

| ![](images/arrSize_streamCompaction.png) |
|:--:|

Using the runtime analysis for scan, we can hence see that the trend persists with the runtime of the entire stream compaction algorithm. CPU runtime will increase exponentially with the increase in array size, while efficient's timing increase remains much faster due to many of the reasons listed above, including but not limited to its optimised memory coalescing, parallel execution of scanning and scattering, and 2*log(n) kernel launches for upsweep and downsweep.

### Size of Array v. Runtime of Radix Sort

**blockSize = 128, array size power of 2**

| Array Size     | Radix  |
|----------------|--------|
| 21             | 65.865 |
| 22 (default)   | 120.236|
| 23             | 189.99 |
| 24             | 338.039|

| ![](images/arrSize_radixSort.png) |
|:--:|

Radix sort uses multiple calls to efficient scan to sort an array based on Least Significant Bit to Most Significant Bit. Because scan is called for every bit and we are sorting with integers (max len 32), the runtime is O(n * k), where k = 32. The arrays b, e, and f which are used in memory as well are also coalesced, contributing to its efficiency. 