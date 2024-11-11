# AmgT

Algebraic Multigrid Solver on Tensor Cores

## Paper

This is the code of our paper published at SC '24:

Yuechen Lu, Lijie Zeng, Tengcheng Wang, Xu Fu, Wenxuan Li, Helin Cheng, Dechuang Yang, Zhou Jin, Marc Casas and Weifeng Liu. 2024. AmgT: Algebraic Multigrid Solver on Tensor Cores. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '24). https://doi.ieeecomputersociety.org/10.1109/SC41406.2024.00058

## Introduction

Algebraic multigrid (AMG) methods are particularly efficient to solve a wide range of sparse linear systems, due to their good flexibility and adaptability. Even though modern parallel devices, such as GPUs, brought massive parallelism to AMG, the latest major hardware features, i.e., tensor core units and their low precision compute power, have not been exploited to accelerate AMG.

This paper proposes AmgT, a new AMG solver that utilizes the tensor core and mixed precision ability of the latest GPUs during multiple phases of the AMG algorithm. Considering that the sparse general matrix-matrix multiplication (SpGEMM) and sparse matrix-vector multiplication (SpMV) are extensively used in the setup and solve phases, respectively, we propose a novel method based on a new unified sparse storage format that leverages tensor cores and their variable precision. Our method improves both the performance of GPU kernels, and also reduces the cost of format conversion in the whole data flow of AMG. To better utilize the algorithm components in existing libraries, the data format and compute kernels of the AmgT solver are incorporated into the HYPRE library. 

The main codes of AmgT are in the following path:

mBSR Data Structure: AmgT\_HYPRE/src/seq\_mv/seq_mv.h

SpGEMM in Setup Phase: AmgT\_HYPRE/src/seq\_mv/csr\_spgemm\_device.c

SpMV in Solve Phase: AmgT\_HYPRE/src/seq\_mv/csr\_matvec\_device.c

## Installation

To better reproduce experiment results, we suggest an NVIDIA GPU with compute capability 8.0. It is best to compile AmgT using CUDA v12.2 or higher verison and OpenMPI v4.0.0 or higher version. 

Change the `CUDA_HOME` and `GPU` in compile.sh according to the specific environment.

Select the compilation version by changing the `execuative` in compile.sh as needed: 
- AmgT_FP64: double-precision AmgT, 
- AmgT_Mixed: mixed-precision AmgT,
- cuSPARSE: double-precision HYPRE that calls the cuSPARSE library.

Then, execute the following command under `AmgT/` to compile the AmgT and test file:  
`$source compile.sh`

## Execution

After the AmgT and test file are compiled, the executable files are in under the path `AmgT/AmgT_test/runnable_files/`. 

Execute the following commands to use the AmgT:
- Use the double-precision AmgT: `./AmgT_FP64 matrix_name.mtx` 
- Use the mixed-precision AmgT:  `./AmgT_Mixed matrix_name.mtx` 
- Use the double-precision HYPRE that calls the cuSPARSE library: `./cuSPARSE matrix_name.mtx`

## Contact us

If you have any questions about running the code, please contact Yuechen Lu.

E-mail: yuechenlu@student.cup.edu.cn
