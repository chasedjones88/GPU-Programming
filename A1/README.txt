Chase Jones

Programs were tested on WEDGE & WICKET in 1013, but neither code run the code due to the Cuda on my home pc being version 10.0 and the lab computers having only version 9.1. Output from 'nvcc --version' on all three computers show below.
Programs would not build in VS 2017 at lab due to access being denied access to template.cu for some reason.
I completely understands if this merits a loss in points; I will download Cuda 9.1 on my home PC and do future projects in Cuda 9.1.

I also included the answers to the questions below and as a pdf in the build directory as ProblemQuestions.pdf

WEDGE & WICKET (Lab 1013) Cuda Vers:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2017 NVIDIA Corporation
Built on Fri_Nov__3_21:08:12_Central_Daylight_Time_2017
Cuda compilation tools, release 9.1, V9.1.85

Home PC Cuda Vers:
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2018 NVIDIA Corporation
Built on Sat_Aug_25_21:08:04_Central_Daylight_Time_2018
Cuda compilation tools, release 10.0, V10.0.130

#########
Problem #1 Questions
#########

1. The maximum number of threads that can be launched is 33,553,920.
2. The programmer may not want to launch the maximum number of threads if they were doing computations on an incomplete block.
3. Lack of data to operate on may limit the program from launching the maximum number of threads, in order to increase computation time spent on other tasks. It'd be silly to launch 2048 threads for something that only required 4. That's 2044 threads that could have been working on something else.
4. Shared memory is memory on the SM (streaming multiprocessor) that can be accessed much faster than global memory. It also allows threads to communicate with one another.
5. Global memory is the general memory of the graphics card, and is much slower than shared memory. Global memory can also cause data hazards if the threads are not synced correctly.
6. Constant memory is memory that will not change over the life of the kernel. This is for things like the grid and block sizes and other constants to be used during kernel launch.
7. Warp size is the number of threads an SM can execute concurrently.

#########
Problem #2 Questions
#########

1. The number of floating point operations being performed by the vector add kernel is simply N additions, since the kernel only really has 2 lines. The first line is an integer addition to get an offset for the float *. The second line is a bit more complex. It does two integer additions to get offsets for the float elements of the float *, then it adds the two floating point operations. The kernel then exits. Therefore, it only executes N floating point operations (C = A+B).
2. The kernel reads 2N times from the global memory, as it has to read every value from one vector then every value from the other vector.
3. The kernel writes N times to the global memory, one time for each value computed. In a vector of size N, it must write N elements.
4. Changing colorspace of a photograph by a linear value for RGB, probabilities in statistics, book-keeping.

#########
Problem #3 Questions
#########

1. For every value in the output matrix, it is a result of (numAColumns^2)*2 floating point operations (multiply value in A by value in B, then add result to Pvalue). Therefore the kernel must perform numCRows*numCColumns*2(numAColumns^2) floating point operations to receive the product.
2. For every value in the output matrix, it is a result of numAColumns^2 global memory reads (numAColums per row/column per input Matrix). Therefore, the kernel must perform numCRows*numCColumns*(numAColumns^2) global memory reads to receive the product.
3. For every value in the output matrix, the kernel must perform numCRows*numCColumns global memory writes.
4. The only way I can think of to optimize the algorithm would possibly being able to store the global memory reads for later use by different threads, so they don't have to be loaded many times over; such as is done in the TiledMatrixMulitiply problem.
5. Transformations in graphical programs, linear algebra, physics calculations.

#########
Problem #3 Questions
#########

1. For every value in the output matrix, it is a result of (numAColumns^2)*2 floating point operations (multiply value in A by value in B, then add result to Pvalue). Therefore the kernel must perform numCRows*numCColumns*2(numAColumns^2) floating point operations to receive the product.
2. For every value in the output matrix, it is a result of (numARows*numAColumns+numAColumns*numBColumns)*((ceil(numAColums/TILE_WIDTH)+1)^2) global reads, since all elements are read into Shared memory once per tile, then used by the other threads accessing the shared memory.
3. For every value in the output matrix, the kernel must perform numCRows*numCColumns global memory writes.
4. Further optimization might require making the shared memory / tiles larger to accomodate more values, and less global memory reads.
5. A lot of errors can be introduced from the tiling aspect: for instance, boundary checking is a must to make sure values aren't being incorrectly set. Another difficulty is using many different variables to reference specific locations, such as all the ty, tx, TILE_WIDTH, Row, Col, and the various boundary sizes.
6. In this case, you'd probably want to split up the memory writes, so some element may only calculate half a row * half a column, and another grid/block take up the computation. Although in this case, you'd want to add to the value in the output matrix, rather than replacing it. Therefore, it's probably a good idea to set all output matrix values to zero first.
7. In this case, it'd be much like the first. You'd have to break the matrices up into smaller pieces which are all initialized to zero, do some calculations, add the values back, and continue with the next grid. It'd be very intensive on the system, but if something can't fit into memory, the only option is to break it up.