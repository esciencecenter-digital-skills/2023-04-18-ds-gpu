![](https://i.imgur.com/iywjz8s.png)


# Collaborative Document Day 2

2023-04-18-ds-gpu

Welcome to The Workshop Collaborative Document.

This Document is synchronized as you type, so that everyone viewing this page sees the same text. This allows you to collaborate seamlessly on documents.

----------------------------------------------------------------------------

## 👮Code of Conduct

Participants are expected to follow these guidelines:
* Use welcoming and inclusive language.
* Be respectful of different viewpoints and experiences.
* Gracefully accept constructive criticism.
* Focus on what is best for the community.
* Show courtesy and respect towards other community members.
 
## ⚖️ License

All content is publicly available under the Creative Commons Attribution License: [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

## 🙋Getting help

To ask a question, just raise your hand.

If you need help from a helper, place a pink post-it note on your laptop lid. A helper will come to assist you as soon as possible.

## 🖥 Workshop websites

* [Workshop](https://esciencecenter-digital-skills.github.io/2023-04-18-ds-gpu/)
* [Course Notes](https://carpentries-incubator.github.io/lesson-gpu-programming/)
* [Google Colab](https://colab.research.google.com/)

## 👩‍🏫👩‍💻🎓 Instructors

Alessio Sclocco, Hanno Spreeuw

## 🧑‍🙋 Helpers

Giulia Crocioni, Laura Ootes

## 🗓️ Agenda
| Time | Topic |
|--:|:---|
| 09:30 | Welcome and icebreaker |
| 10:00 | Introduction to CUDA |
| 10:30 | Coffee break |
| 10:45 | CUDA memories and their use |
| 11:15 | Coffee break |
| 11:30 | CUDA memories and their use |
| 12:00 | Lunch break |
| 13:00 | Data sharing and synchronization |
| 14:00 | Coffee break |
| 14:15 | Data sharing and synchronization |
| 15:00 | Coffee break |
| 15:15 | Concurrent access to the GPU |
| 16:15 | Wrap-up |
| 16:30 | End |


## 🏢 Location logistics
* Coffee and toilets are in the hallway, just outside of the classroom.
* If you leave the building, 
  be sure to be accompanied by someone from the escience center to let you back in through the groundfloor door
* For access to this floor you might need to ring the doorbell so someone can let you in
* In case of an emergency, you can exit our floor using the main staircase.
  Or follow green light signs at the ceiling to the emergency staircase.
* **Wifi**: Eduroam should work. Otherwise use the 'matrixbuilding' network, password should be printed out and available somewhere in the room.

## 🎓 Certificate of attendance
If you attend the full workshop you can request a certificate of attendance by emailing to training@esciencecenter.nl .

## 🔧 Exercises

### Exercise 1: Scaling up

In the following code, fill in the blank to work with vectors that are larger than the largest CUDA block (i.e. 1024).

```c
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = ______________;
   C[item] = A[item] + B[item];
}
```

### Solution 1

The correct answer is `(blockIdx.x * blockDim.x) + threadIdx.x`.

```python=
import cupy as cp

size = 2048

# allocate memory
a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

# cuda code
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   C[item] = A[item] + B[item];
}
'''

# compile
vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((2, 1, 1), (size // 2, 1, 1,), (a_gpu, b_gpu, c_gpu, size))
```

### Exercise 2: Compute prime numbers with CUDA

Given the following Python code, similar to what we have seen in the previous episode about Numba, write the missing CUDA kernel that computes all the prime numbers up to a certain upper bound.

```python
import numpy as np
import cupy
import math
from cupyx.profiler import benchmark

# CPU version
def all_primes_to(upper : int, prime_list : list):
    for num in range(0, upper):
        prime = True
        for i in range(2, (num // 2) + 1):
            if (num % i) == 0:
                prime = False
                break
        if prime:
            prime_list[num] = 1

upper_bound = 100_000
all_primes_cpu = np.zeros(upper_bound, dtype=np.int32)

# GPU version
check_prime_gpu_code = r'''
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
   for ( int number = 0; number < size; number++ )
   {
       int result = 1;
       for ( int factor = 2; factor <= number / 2; factor++ )
       {
           if ( number % factor == 0 )
           {
               result = 0;
               break;
           }
       }

       all_prime_numbers[number] = result;
   }
}
'''
# Allocate memory
all_primes_gpu = cupy.zeros(upper_bound, dtype=cupy.int32)

# Setup the grid
all_primes_to_gpu = cupy.RawKernel(check_prime_gpu_code, "all_primes_to")
grid_size = (int(math.ceil(upper_bound / 1024)), 1, 1)
block_size = (1024, 1, 1)

# Benchmark and test
%timeit -n 1 -r 1 all_primes_to(upper_bound, all_primes_cpu)
execution_gpu = benchmark(all_primes_to_gpu, (grid_size, block_size, (upper_bound, all_primes_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")

if np.allclose(all_primes_cpu, all_primes_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

There is no need to modify anything in the code, except writing the body of the CUDA `all_primes_to` inside the `check_prime_gpu_code` string, as we did in the examples so far.

Be aware that the provided CUDA code is a direct port of the Python code, and therefore **very** slow. If you want to test it, user a lower value for `upper_bound`.


### Solution 2

One possible solution for the CUDA kernel is provided in the following code.

```C
extern "C"
__global__ void all_primes_to(int size, int * const all_prime_numbers)
{
    int number = (blockIdx.x * blockDim.x) + threadIdx.x;
    int result = 1;

    if ( number < size )
    {
        for ( int factor = 2; factor <= number / 2; factor++ )
        {
            if ( number % factor == 0 )
            {
                result = 0;
                break;
            }
        }

        all_prime_numbers[number] = result;
    }
}
```

The outermost loop in Python is replaced by having each thread testing for primeness a different number of the sequence. Having one number assigned to each thread via its ID, the kernel implements the innermost loop the same way it is implemented in Python.

### Exercise 3: use shared memory to speed up the histogram

Implement a new version of the CUDA `histogram` function that uses shared memory to reduce conflicts in global memory.
Modify the following code and follow the suggestions in the comments.

```c
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Declare temporary histogram in shared memory
    int temp_histogram[256];
    
    // Update the temporary histogram in shared memory
    atomicAdd();
    // Update the global histogram in global memory, using the temporary histogram
    atomicAdd();
}


```

Hint: for this exercise, you can safely assume that the size of `output` is the same as the number of threads in a block.

Hint: `atomicAdd` can be used on both global and shared memory.

### Solution 3

The following code shows one of the possible solutions.

```python=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];

    atomicAdd(&(temp_histogram[input[item]]), 1);
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```

### Exercise 4: parallel reduction

Modify the parallel reduction CUDA kernel and make it work.

Kernel:

```python
cuda_code = r'''
#define block_size_x 256

extern "C"
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    //cooperatively (with all threads in all thread blocks) iterate over input array
    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }

    //at this point we have reduced the number of values to be summed from n to
    //the total number of threads in all thread blocks combined

    //the goal is now to reduce the values within each thread block to a single
    //value per thread block for this we will need shared memory

    //declare shared memory array, how much shared memory do we need?
    //__shared__ float ...;

    //make every thread store its thread-local sum to the array in shared memory
    //... = sum;
    
    //now let's call syncthreads() to make sure all threads have finished
    //storing their local sums to shared memory
    __syncthreads();

    //now this interesting looking loop will do the following:
    //it iterates over the block_size_x with the following values for s:
    //if block_size_x is 256, 's' will be powers of 2 from 128, 64, 32, down to 1.
    //these decreasing offsets can be used to reduce the number
    //of values within the thread block in only a few steps.
    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {

        //you are to write the code inside this loop such that
        //threads will add the sums of other threads that are 's' away
        //do this iteratively such that together the threads compute the
        //sum of all thread-local sums 

        //use shared memory to access the values of other threads
        //and store the new value in shared memory to be used in the next round
        //be careful that values that should be read are
        //not overwritten before they are read
        //make sure to call __syncthreads() when needed
    }

    //write back one value per thread block
    if (ti == 0) {
        //out_array[blockIdx.x] = ;  //store the per-thread block reduced value to global memory
    }
}
'''
```

Python host code, it does not need to be modified for the exercise.

```python
import numpy
# Allocate memory
size = numpy.int32(5e7)
input_cpu = numpy.random.randn(size).astype(numpy.float32) + 0.00000001
input_gpu = cupy.asarray(input_cpu)
out_gpu = cupy.zeros(2048, dtype=cupy.float32)

# Compile CUDA kernel
grid_size = (2048, 1, 1)
block_size = (256, 1, 1)
reduction_gpu = cupy.RawKernel(cuda_code, "reduce_kernel")

# Execute athe first partial reduction
reduction_gpu(grid_size, block_size, (out_gpu, input_gpu, size))
# Execute the second and final reduction
reduction_gpu((1, 1, 1), block_size, (out_gpu, out_gpu, 2048))

# Execute and time CPU code
sum_cpu = numpy.sum(input_cpu)

if numpy.absolute(sum_cpu - out_gpu[0]) < 1.0:
    print("Correct results!")
else:
    print("Wrong results!")
```

### Solution 4

To have working code, it is just necessary to modify the CUDA kernel in the following way.

```python
cuda_code = r'''
#define block_size_x 256

extern "C"
__global__ void reduce_kernel(float *out_array, float *in_array, int n) {

    int ti = threadIdx.x;
    int x = blockIdx.x * block_size_x + threadIdx.x;
    int step_size = gridDim.x * block_size_x;
    float sum = 0.0f;

    for (int i=x; i<n; i+=step_size) {
        sum += in_array[i];
    }
    
    __shared__ float temp_sum[block_size_x];

    temp_sum[ti] = sum;
    
    __syncthreads();

    #pragma unroll
    for (unsigned int s=block_size_x/2; s>0; s/=2) {
        if ( ti < s )
        {
            temp_sum[ti] += temp_sum[ti + s];
        }
        __syncthreads();
    }

    if (ti == 0) {
        out_array[blockIdx.x] = temp_sum[0];
    }
}
'''
```

## 🧠 Collaborative Notes

### Your First GPU Kernel

#### Summing Two Vectors in Python

```python=
def vector_add(A, B, C, size):
    for item in range(0, size):
        C[item] = A[item] + B[item]
```

One of the characteristics of this program is that each iteration of the `for` loop is independent from the other iterations. In other words, we could reorder the iterations and still produce the same output, or even compute each iteration in parallel or on a different device, and still come up with the same output. These are the kind of programs that we would call _naturally parallel_, and they are perfect candidates for being executed on a GPU.


#### Summing Two Vectors in CUDA

The CUDA-C language is a GPU programming language and API developed by NVIDIA. It is mostly equivalent to C/C++, with some special keywords, built-in variables, and functions.

We begin our introduction to CUDA by writing a small kernel, i.e. a GPU program, that computes the same function that we just described in Python.

```C
extern "C"
__global__ void vector_add(const float * A, const float * B, const float * C, const int size) {
    int item = threadIdx.x;
    C[item] = A [item] + B[item];
}
```

#### Running Code on the GPU with CuPy

```python=
import cupy as cp

size = 1024

# allocate memory
a_gpu = cp.random.rand(size, dtype=cp.float32)
b_gpu = cp.random.rand(size, dtype=cp.float32)
c_gpu = cp.zeros(size, dtype=cp.float32)

# cuda code
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size) {
    int item = threadIdx.x;
    C[item] = A [item] + B[item];
}
'''

# compile
vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1,), (a_gpu, b_gpu, c_gpu, size))
```

To be sure that the CUDA code does exactly what we want, we can execute our sequential Python code and compare the results.

```python=
import numpy as np

a_cpu = cp.asnumpy(a_gpu)
b_cpu = cp.asnumpy(b_gpu)
c_cpu = np.zeros(size, dtype=np.float32)

vector_add(a_cpu, b_cpu, c_cpu, size)

np.allclose(c_cpu, c_gpu)
```

#### Understanding the CUDA Code

What is going on?

![](https://codimd.carpentries.org/uploads/upload_108416b9598363bfe4843b94499308e5.png)

`threadIdx` = ID of a thread inside a block

`blockDim` = the size of a block == the number of threads per dimension

`blockIdx` = ID of a block

`gridDim` = the size of the grid == the number of blocks per dimension

In the previous example we had a small vector of size 1024, and each of the 1024 threads we generated was working on one of the element.

What would happen if we changed the size of the vector to a larger number, such as 2048? We modify the value of the variable size and try again.

```python=
# size of the vectors
size = 2048

# allocating and populating the vectors
a_gpu = cupy.random.rand(size, dtype=cupy.float32)
b_gpu = cupy.random.rand(size, dtype=cupy.float32)
c_gpu = cupy.zeros(size, dtype=cupy.float32)

# CUDA vector_add
vector_add_gpu = cupy.RawKernel(r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
    int item = threadIdx.x;
    C[item] = A[item] + B[item];
}
''', "vector_add")

vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

This is how the output should look like when running the code in a Jupyter Notebook:

```
---------------------------------------------------------------------------

CUDADriverError                           Traceback (most recent call last)

<ipython-input-4-a26bc8acad2f> in <module>()
     19 ''', "vector_add")
     20 
---> 21 vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
     22 
     23 print(c_gpu)

cupy/core/raw.pyx in cupy.core.raw.RawKernel.__call__()

cupy/cuda/function.pyx in cupy.cuda.function.Function.__call__()

cupy/cuda/function.pyx in cupy.cuda.function._launch()

cupy_backends/cuda/api/driver.pyx in cupy_backends.cuda.api.driver.launchKernel()

cupy_backends/cuda/api/driver.pyx in cupy_backends.cuda.api.driver.check_status()

CUDADriverError: CUDA_ERROR_INVALID_VALUE: invalid argument
```

The reason for this error is that most GPUs will not allow us to execute a block composed of more than 1024 threads. If we look at the parameters of our functions we see that the first two parameters are two triplets. 

```python=
vector_add_gpu((1, 1, 1), (size, 1, 1), (a_gpu, b_gpu, c_gpu, size))
```

The first triplet specifies the size of the CUDA **grid**, while the second triplet specifies the size of the CUDA **block**. The grid is a three-dimensional structure in the CUDA programming model and it represent the organization of a whole kernel execution. A grid is made of one or more independent blocks, and in the case of our previous snippet of code we have a grid composed by a single block `(1, 1, 1)`. The size of this block is specified by the second triplet, in our case `(size, 1, 1)`. While blocks are independent of each other, the thread composing a block are not completely independent, they share resources and can also communicate with each other.

To go back to our example, we can modify che grid specification from `(1, 1, 1)` to `(2, 1, 1)`, and the block specification from `(size, 1, 1)` to `(size // 2, 1, 1)`. If we run the code again, we should now get the expected output.

#### Vectors of Arbitrary Size

So far we have worked with a number of threads that is the same as the elements in the vector. However, in a real world scenario we may have to process vectors of arbitrary size, and to do this we need to modify both the kernel and the way it is launched.

```python=
vector_add_cuda_code = r'''
extern "C"
__global__ void vector_add(const float * A, const float * B, float * C, const int size)
{
   int item = (blockIdx.x * blockDim.x) + threadIdx.x;
   if ( item < size )
   {
      C[item] = A[item] + B[item];
   }
}
'''
vector_add_gpu = cp.RawKernel(vector_add_cuda_code, "vector_add")

threads_per_block = 1024
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

vector_add_gpu(grid_size, block_size, (a_gpu, b_gpu, c_gpu, size))

if np.allclose(c_cpu, c_gpu):
    print("Correct results!")
else:
    print("Wrong results!")
```

### Registers, Global, and Local Memory

#### Registers

Registers are fast on-chip memories that are used to store operands for the operations executed by the computing cores.

Did we encounter registers in the `vector_add` code used in the previous episode? Yes we did! The variable `item` is, in fact, stored in a register for at least part, if not all, of a thread’s execution. In general all scalar variables defined in CUDA code are stored in registers.

Registers are local to a thread, and each thread has exclusive access to its own registers: values in registers cannot be accessed by other threads, even from the same block, and are not available for the host. Registers are also not permanent, therefore data stored in registers is only available during the execution of a thread.

#### Global memory

Global memory can be considered the main memory space of the GPU in CUDA. It is allocated, and managed, by the host, and it is accessible to both the host and the GPU, and for this reason the global memory space can be used to exchange data between the two. It is the largest memory space available, and therefore it can contain much more data than registers, but it is also slower to access. This memory space does not require any special memory space identifier.

### Shared Memory and Synchronization

#### Shared Memory

Shared memory is a CUDA memory space that is shared by all threads in a thread block. In this case shared means that all threads in a thread block can write and read to block-allocated shared memory, and all changes to this memory will be eventually available to all threads in the block.

To allocate an array in shared memory we need to preface the definition with the identifier `__shared__`.

Let us work on an example where using shared memory is actually useful. We start with some Python code.

```python=
def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1
```

The `histogram` function, as the name suggests, computes the histogram of an array of integers, i.e. counts how many instances of each integer are in `input_array`, and writes the count in `output_array`. We can now generate some data and run the code.

```python=
input_array = np.random.randint(256, size=2048, dtype=np.int32)
output_array = np.zeros(256, dtype=np.int32)
histogram(input_array, output_array)
```

Everything as expected. We can now write the equivalent code in CUDA.

```C=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    output[input[item]] = output[input[item]] + 1;
}
```

The GPU is a highly parallel device, executing multiple threads at the same time. In the previous code different threads could be updating the same output item at the same time, producing wrong results.

To solve this problem, we need to use a function from the CUDA library named `atomicAdd`. This function ensures that the increment of `output_array` happens in an atomic way, so that there are no conflicts in case multiple threads want to update the same item at the same time.

```C=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
```

And the full Python code snippet.

```python=
import math
import numpy as np
import cupy
from cupyx.profiler import benchmark

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1

# input size
size = 2**25

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;

    atomicAdd(&(output[input[item]]), 1);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
if np.allclose(output_cpu, output_gpu):
    print("Correct results!")
else:
    print("Wrong results!")

# measure performance
%timeit -n 1 -r 1 histogram(input_cpu, output_cpu)
execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")
```

#### Thread Synchronization

There is still one potentially big issue in the `histogram` code we just wrote, and the issue is that shared memory is not coherent without explicit synchronization. The problem lies in the following two lines of code:

```python=
atomicAdd(&(temp_histogram[input[item]]), 1);
atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
```

In the first line each thread updates one arbitrary position in shared memory, depending on the value of the input, while in the second line each thread reads the element in shared memory corresponding to its thread ID. However, the changes to shared memory are not automatically available to all other threads, and therefore the final result may not be correct.

To solve this issue, we need to explicitly synchronize all threads in a block, so that memory operations are also finalized and visible to all. To synchronize threads in a block, we use the `__syncthreads()` CUDA function. Moreover, shared memory is not initialized, and the programmer needs to take care of that too. So we need to first initialize `temp_histogram`, wait that all threads are done doing this, perform the computation in shared memory, wait again that all threads are done, and only then update the global array.

```C=
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
```

And the full Python code snippet.

```python=
import math
import numpy as np
import cupy
from cupyx.profiler import benchmark

def histogram(input_array, output_array):
    for item in input_array:
        output_array[item] = output_array[item] + 1

# input size
size = 2**25

# allocate memory on CPU and GPU
input_gpu = cupy.random.randint(256, size=size, dtype=cupy.int32)
input_cpu = cupy.asnumpy(input_gpu)
output_gpu = cupy.zeros(256, dtype=cupy.int32)
output_cpu = cupy.asnumpy(output_gpu)

# CUDA code
histogram_cuda_code = r'''
extern "C"
__global__ void histogram(const int * input, int * output)
{
    int item = (blockIdx.x * blockDim.x) + threadIdx.x;
    __shared__ int temp_histogram[256];
 
    // Initialize shared memory and synchronize
    temp_histogram[threadIdx.x] = 0;
    __syncthreads();

    // Compute shared memory histogram and synchronize
    atomicAdd(&(temp_histogram[input[item]]), 1);
    __syncthreads();

    // Update global histogram
    atomicAdd(&(output[threadIdx.x]), temp_histogram[threadIdx.x]);
}
'''

# compile and setup CUDA code
histogram_gpu = cupy.RawKernel(histogram_cuda_code, "histogram")
threads_per_block = 256
grid_size = (int(math.ceil(size / threads_per_block)), 1, 1)
block_size = (threads_per_block, 1, 1)

# check correctness
histogram(input_cpu, output_cpu)
histogram_gpu(grid_size, block_size, (input_gpu, output_gpu))
if np.allclose(output_cpu, output_gpu):
    print("Correct results!")
else:
    print("Wrong results!")

# measure performance
%timeit -n 1 -r 1 histogram(input_cpu, output_cpu)
execution_gpu = benchmark(histogram_gpu, (grid_size, block_size, (input_gpu, output_gpu)), n_repeat=10)
gpu_avg_time = np.average(execution_gpu.gpu_times)
print(f"{gpu_avg_time:.6f} s")
```

### Concurrent access to the GPU

#### Concurrently execute two kernels on the same GPU

So far we only focused on completing one operation at the time on the GPU, writing and executing a single CUDA kernel each time. However the GPU has enough resources to perform more than one task at the same time.

In a real application we may want to run the two kernels concurrently on the GPU to reduce the total execution time. To achieve this in CUDA we need to use CUDA _streams_.

A stream is a sequence of GPU operations that is executed in order, and so far we have been implicitly using the defaul stream. This is the reason why all the operations we issued, such as data transfers and kernel invocations, are performed in the order we specify them in the Python code, and not in any other.

Have you wondered why after requesting data transfers to and from the GPU, we do not stop to check if they are complete before performing operations on such data? The reason is that within a stream all operations are carried out in order, so the kernel calls in our code are always performed after the data transfer from host to device is complete, and so on.

If we want to create new CUDA streams, we can do it this way using CuPy.

```python=
stream_one = cp.cuda.Stream()
stream_two = cp.cuda.Stream()
```

We can then execute the kernels in different streams by using the Python `with` statement.

```python=
with stream_one:
    gpu_function_one()
with stream_two:
    gpu_function_two()
```

Using the `with` statement we implicitly execute the CUDA operations in the code block using that stream. The result of doing this is that the second kernel, i.e. `histogram_gpu`, does not need to wait for `all_primes_to_gpu` to finish before being executed.

#### Stream synchronization

If we need to wait for all operations on a certain stream to finish, we can call the `synchronize` method. Continuing with the previous example, in the following Python snippet we wait for the execution of `gpu_function_one` on `stream_one` to finish.

```python=
stream_one.synchronize()
```

This synchronization primitive is useful when we need to be sure that all operations on a stream are finished, before continuing. It is, however, a bit coarse grained. Imagine to have a stream with a whole sequence of operations enqueued, and another stream with one data dependency on one of these operations. If we use `synchronize`, we wait until all operations of said stream are completed before executing the other stream, thus negating the whole reason of using streams in the first place.

A possible solution is to insert a CUDA _event_ at a certain position in the stream, and then wait specifically for that event. Events are created in Python in the following way.

```python=
interesting_event = cupy.cuda.Event()
```

## :question: Questions

## 📚 Resources

* [CodiMD Markdown Guide](https://www.markdownguide.org/tools/codimd/)
* [CuPy User Guide](https://docs.cupy.dev/en/stable/user_guide/index.html)
* [CUDA Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
