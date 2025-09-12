CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jacqueline (Jackie) Li
  * [LinkedIn](https://www.linkedin.com/in/jackie-lii/), [personal website](https://sites.google.com/seas.upenn.edu/jacquelineli/home), [Instagram](https://www.instagram.com/sagescherrytree/), etc.
* Tested on: Windows 10, 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, NVIDIA GeForce RTX 3060 Laptop GPU (6 GB)

# Program Overview

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

  For each element in parallel:
    b[i] = (idata[i] >> bit) & 1
    e[i] = 1 - b[i]

  Run exclusive scan on f

  totalFalse = total numnber of zeroes

  For each element in parallel, i is index:
    t[i] = i - f[i] + totalFalse 

  For each element in parallel, i is index:
    if b[i] == 0:
       odata[f[i]] = idata[i]
    else:
       odata[t[i]] = idata[i]

  Swap(idata, odata)
```

Runtime: O(n) or O(n * 32)

## Runtime Analysis

