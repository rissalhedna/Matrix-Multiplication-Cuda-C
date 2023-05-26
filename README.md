# Matrix Multiplication
Matrix multiplication, also known as matrix product and the multiplication of two matrices, produces a single matrix. It is a type of binary operation.
If A and B are the two matrices, then the product of the two matrices A and B are denoted by:
X = AB

## Parallelizing Matrix Multiplication
In order to optimize the performance of the matrix multiplication algorithm, we make use of the GPU's parallel structure in order to execute different independent components of the multiplication with different threads. This way, each thread will be able to perform its own dot product (shown above), allowing each of them to execute independently of the others.
In order to parallelize the process, we adapt two different methods, the first without using tiling and the second using it. Then, we will have the ability to compare the performance of both methods to choose the optimal one.
Our initial hypothesis is that tiling will substantially increase performance and speedup factor because of its ability to reduce memory access latency and provide a buffer in shared memory for fast access from threads.

### Without Tiling
1. Kernel function:
In our kernel function, each thread will execute a dot product using one row from matrix M and a column from matrix N. Since each dot product in the matrix multiplication is independent, each thread will be able to execute freely without the need for synchronization between the threads.

2. Grid/Block Dimensions:
In our host, we define the number of blocks per grid as well as the number of threads per block before calling our kernel function and executing it on the device. In this specific example, we chose a block size of 32x32 for our tests, meaning that each block contains 512 threads.

The blocks are then tasked with processing a part of the matrix multiplication each, further distributing the job on their threads according to the row/column that they encounter.

Each thread accesses the memory in order to extract the rows and columns from matrices A and B, then calculate the dot product between them. For the threads to access the right values, we used their indices to indicate which rows/columns they are tasked with computing the product for.

## Speedup factor and efficiency:
We define the speedup factor by the following equation:
S(p) = Serial Time / Parallel Time

And the efficiency as follows:
E = S(p) / N_threads

## With Tiling
Due to the high number of memory operations that threads must perform to compute matrix multiplication, there is a latency that gets added to the program's execution, which will negatively affect its performance. For that reason, tiling is used, which is a method that consists of dividing the data into tiles that get loaded individually into the shared memory, which has high access speed, on the contrary to the global memory.

## Setting up the tiles:

Synchronization:
The __synchthreads() function allows us to make sure that before the tiling operation occurs, all threads are ready and that all threads are done with their job before moving on to the next section of the code.

In our code, we used a tile size of 16 and tested our code on all previous block sizes to compare the two methods. We got the following results:

## Discussion
This study investigated how tiling can improve performance compared to CUDA without tiling. Tiling means dividing the input matrices into smaller parts, which are processed separately, reducing the need to access global memory and increasing data reuse through shared memory.
The results consistently showed that using tiling with CUDA was better than not using tiling. It made the computation faster, with significantly lower execution times for all tile sizes.
Using tiling also achieved higher speedup, meaning the computation was even faster compared to not using tiling. It made better use of the GPU's ability to process things in parallel.
Tiling also made better use of the GPU's resources and improved efficiency in terms of how the workload was distributed and how threads were used.
In conclusion, tiling improved performance by reducing execution time, increasing speed, and improving efficiency compared to not using tiling. Tiling is a useful technique for making matrix multiplication on GPUs much faster.
