/******************************************************************************
 * Copyright (c) 1998 Lawrence Livermore National Security, LLC and other
 * HYPRE Project Developers. See the top-level COPYRIGHT file for details.
 *
 * SPDX-License-Identifier: (Apache-2.0 OR MIT)
 ******************************************************************************/

/******************************************************************************
 *
 * Matvec functions for hypre_CSRMatrix class.
 *
 *****************************************************************************/

#include "seq_mv.h"
#include "_hypre_utilities.hpp"
#include "seq_mv.hpp"

double time_spmv_preprocess = 0;
double time_spmv_sum = 0;
int spmv_times = 0;

double csr2bsr_step1 = 0;
double csr2bsr_step2 = 0;
double csr2bsr_step3 = 0;

#if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

// #if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
// #define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_CSR_ALG2
// #define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG3

#elif CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_SPMM_CSR_ALG1

#else
#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
#define HYPRE_CUSPARSE_SPMM_ALG CUSPARSE_CSRMM_ALG1
#endif

// #define gettimeofday1(a, b) \ 
//         cudaDeviceSynchronize(); \
//         gettimeofday(a, b)

/* y = alpha * A * x + beta * y
 * This function is supposed to be only used inside the other functions in this file
 */
static inline HYPRE_Int
hypre_CSRMatrixMatvecDevice2(HYPRE_Int trans,
                             HYPRE_Complex alpha,
                             hypre_CSRMatrix *A,
                             hypre_Vector *x,
                             HYPRE_Complex beta,
                             hypre_Vector *y,
                             HYPRE_Int offset)
{
    /* Sanity check */
    if (hypre_VectorData(x) == hypre_VectorData(y))
    {
        hypre_error_w_msg(HYPRE_ERROR_GENERIC,
                          "ERROR::x and y are the same pointer in hypre_CSRMatrixMatvecDevice2");
    }

#if defined(HYPRE_USING_CUSPARSE) ||  \
    defined(HYPRE_USING_ROCSPARSE) || \
    defined(HYPRE_USING_ONEMKLSPARSE)

    /* Input variables */
    HYPRE_Int num_vectors_x = hypre_VectorNumVectors(x);
    HYPRE_Int num_vectors_y = hypre_VectorNumVectors(y);

    /* Local variables */
    HYPRE_Int use_vendor = hypre_HandleSpMVUseVendor(hypre_handle());

#if defined(HYPRE_USING_CUSPARSE) && CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
    HYPRE_Int multivec_storage_x = hypre_VectorMultiVecStorageMethod(x);
    HYPRE_Int multivec_storage_y = hypre_VectorMultiVecStorageMethod(y);

    /* Force use of hypre's SpMV for row-wise multivectors */
    if ((num_vectors_x > 1 && multivec_storage_x == 1) ||
        (num_vectors_y > 1 && multivec_storage_y == 1))
    {
        use_vendor = 0;
    }
#else
    /* TODO - enable cuda 10, rocsparse, and onemkle sparse support for multi-vectors */
    if (num_vectors_x > 1 || num_vectors_y > 1)
    {
        use_vendor = 0;
    }
#endif

    if (use_vendor)
    {
#if defined(HYPRE_USING_CUSPARSE)
        hypre_CSRMatrixMatvecCusparse(trans, alpha, A, x, beta, y, offset);

#elif defined(HYPRE_USING_ROCSPARSE)
        hypre_CSRMatrixMatvecRocsparse(trans, alpha, A, x, beta, y, offset);

#elif defined(HYPRE_USING_ONEMKLSPARSE)
        hypre_CSRMatrixMatvecOnemklsparse(trans, alpha, A, x, beta, y, offset);
#endif
    }
    else
#endif // defined(HYPRE_USING_CUSPARSE) || defined(HYPRE_USING_ROCSPARSE) ...
    {
#if defined(HYPRE_USING_GPU)
        hypre_CSRMatrixSpMVDevice(trans, alpha, A, x, beta, y, 0);

#elif defined(HYPRE_USING_DEVICE_OPENMP)
        hypre_CSRMatrixMatvecOMPOffload(trans, alpha, A, x, beta, y, offset);
#endif
    }

    return hypre_error_flag;
}

/* y = alpha * A * x + beta * b */
HYPRE_Int
hypre_CSRMatrixMatvecDevice(HYPRE_Int trans,
                            HYPRE_Complex alpha,
                            hypre_CSRMatrix *A,
                            hypre_Vector *x,
                            HYPRE_Complex beta,
                            hypre_Vector *b,
                            hypre_Vector *y,
                            HYPRE_Int offset)
{
    HYPRE_Int m_a = hypre_CSRMatrixNumRows(A);
    // hypre_GpuProfilingPushRange("CSRMatrixMatvec");
    HYPRE_Int num_vectors = hypre_VectorNumVectors(x);

    // TODO: RL: do we need offset > 0 at all?
    hypre_assert(offset == 0);

    // VPM: offset > 0 does not work with multivectors. Remove offset? See comment above
    hypre_assert(!(offset != 0 && num_vectors > 1));
    hypre_assert(num_vectors > 0);

    HYPRE_Int nx = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
    HYPRE_Int ny = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);

    // RL: Note the "<=", since the vectors sometimes can be temporary work spaces that have
    //     large sizes than the needed (such as in par_cheby.c)
    hypre_assert(ny <= hypre_VectorSize(y));
    hypre_assert(nx <= hypre_VectorSize(x));
    hypre_assert(ny <= hypre_VectorSize(b));

    // hypre_CSRMatrixPrefetch(A, HYPRE_MEMORY_DEVICE);
    // hypre_SeqVectorPrefetch(x, HYPRE_MEMORY_DEVICE);
    // hypre_SeqVectorPrefetch(b, HYPRE_MEMORY_DEVICE);
    // if (hypre_VectorData(b) != hypre_VectorData(y))
    //{
    //    hypre_SeqVectorPrefetch(y, HYPRE_MEMORY_DEVICE);
    // }

    if (hypre_VectorData(b) != hypre_VectorData(y))
    {
        hypre_TMemcpy(hypre_VectorData(y) + offset,
                      hypre_VectorData(b) + offset,
                      HYPRE_Complex,
                      (ny - offset) * num_vectors,
                      hypre_VectorMemoryLocation(y),
                      hypre_VectorMemoryLocation(b));
    }

    if (hypre_CSRMatrixNumNonzeros(A) <= 0 || alpha == 0.0)
    {
        hypre_SeqVectorScale(beta, y);
    }
    else
    {
        hypre_CSRMatrixMatvecDevice2(trans, alpha, A, x, beta, y, offset);
    }

#if defined(HYPRE_USING_GPU)
    hypre_SyncComputeStream(hypre_handle());
#endif

    // hypre_GpuProfilingPopRange();

    return hypre_error_flag;
}

#if defined(HYPRE_USING_CUSPARSE)
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

/*--------------------------------------------------------------------------
 * hypre_CSRMatrixMatvecCusparseNewAPI
 *
 * Sparse Matrix/(Multi)Vector interface to cusparse's API 11
 *
 * Note: The descriptor variables are not saved to allow for generic input
 *--------------------------------------------------------------------------*/

__device__ __forceinline__ void mma_m16n8k4_tf32_spmv(float *acc, uint32_t *frag_a, uint32_t *frag_b)
{
    asm volatile(

        "mma.sync.aligned.m16n8k4.row.col.f32.tf32.tf32.f32"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6 }, "
        " { %0, %1, %2, %3 };"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(frag_a[0]), "r"(frag_a[1]),
          "r"(frag_b[0]));
}
__device__ __forceinline__ void mma_m8n8k4_fp16(half *acc, uint32_t *A, uint32_t *B)
{
    uint32_t *C = reinterpret_cast<uint32_t *>(&acc[0]);

    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16"
        " { %0, %1, %2, %3 }, "
        " { %4, %5 }, "
        " { %6, %7 }, "
        " { %0, %1, %2, %3 };"
        : "+r"(C[0]), "+r"(C[1]), "+r"(C[2]), "+r"(C[3]) : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));
}

__forceinline__ __device__ int sum_warp_shfl_int(int sum)
{
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffff, sum, mask);

    return sum;
}

int BinarySearch(int *arr, int len, int target)
{
    int low = 0;
    int high = len;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__device__ __host__ int BinarySearch2(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}

__device__ __host__ int BinarySearch3(unsigned int *arr, int left, int right, unsigned int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__device__ __forceinline__ void mma_m8n8k4(MAT_VAL_TYPE *acc, MAT_VAL_TYPE &frag_a, MAT_VAL_TYPE &frag_b)
{
    asm volatile(
        "mma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64"
        " { %0, %1 }, "
        " { %2 }, "
        " { %3 }, "
        " { %0, %1 };"
        : "+d"(acc[0]), "+d"(acc[1]) : "d"(frag_a), "d"(frag_b));
}

__device__ __host__ int BinarySearch2_SpMV(int *arr, int left, int right, int target)
{
    int low = left;
    int high = right;
    int mid = 0;
    while (low <= high)
    {
        mid = (low + high) / 2;
        if (target < arr[mid])
            high = mid - 1;
        else if (target > arr[mid])
            low = mid + 1;
        else
            return mid;
    }
    return -1;
}
__global__ void bsr_spmv(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                         MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                         int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha, MAT_VAL_TYPE beta)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[0] + beta * d_y[rowid];
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] = alpha * fragC[1] + beta * d_y[rowid];
    }
}

void blcMat_cpy_H2D(bsrMAT *d_mat, bsrMAT *h_mat)
{
    d_mat->row = h_mat->row;
    d_mat->col = h_mat->col;
    d_mat->nnz = h_mat->nnz;
    d_mat->blc_row = h_mat->blc_row;
    d_mat->blc_col = h_mat->blc_col;
    d_mat->blc_num = h_mat->blc_num;

    cudaMalloc((void **)&(d_mat->blcPtr), sizeof(MAT_PTR_TYPE) * (d_mat->blc_row + 1));
    cudaMalloc((void **)&(d_mat->blcIdx), sizeof(MAT_IDX_TYPE) * d_mat->blc_num);
    cudaMalloc((void **)&(d_mat->blcVal), sizeof(MAT_VAL_TYPE) * d_mat->nnz);
    cudaMalloc((void **)&(d_mat->blcMap), sizeof(MAT_MAP_TYPE) * d_mat->blc_num);

    cudaMemcpy(d_mat->blcPtr, h_mat->blcPtr, sizeof(MAT_PTR_TYPE) * (d_mat->blc_row + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcIdx, h_mat->blcIdx, sizeof(MAT_IDX_TYPE) * d_mat->blc_num, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcVal, h_mat->blcVal, sizeof(MAT_VAL_TYPE) * d_mat->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat->blcMap, h_mat->blcMap, sizeof(MAT_MAP_TYPE) * d_mat->blc_num, cudaMemcpyHostToDevice);
}

void release_host_bsrMAT(bsrMAT mat)
{
    free(mat.blcPtr);
    free(mat.blcIdx);
    free(mat.blcMap);
    free(mat.blcVal);
}

__global__ void bsr_spmv_balanced_cc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] += fragC[0] * alpha;
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] += fragC[1] * alpha;
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] += fragC[0] * alpha;
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] += fragC[1] * alpha;
    }
}

__global__ void bsr_spmv_cc_fp64(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, MAT_VAL_TYPE *d_blcVal,
                                 MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y,
                                 int blc_row, int blc_col, int row, int col, MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y[blc_rid * BSR_M + laneid] += alpha * res;
    }
}
__global__ void bsr_spmv_balanced_tc_fp64(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_VAL_TYPE *d_blcVal,
                                          MAT_VAL_TYPE *d_x, MAT_VAL_TYPE *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;

    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    MAT_VAL_TYPE fragA, fragB, fragC[2] = {0};
    for (int i = start; i < end; i += 2)
    {
        MAT_VAL_TYPE *cur_val = d_blcVal + i * BSR_NNZ;
        fragA = (i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid];

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        fragB = d_x[xid + laneid_mod_4];

        mma_m8n8k4(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}

__global__ void get_rowPtrbyWarp(MAT_PTR_TYPE *d_blcPtr, int *rowPtrbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    rowPtrbyWarp[rowid] = (d_blcPtr[rowid + 1] - d_blcPtr[rowid] + WARP_CAPACITY - 1) / WARP_CAPACITY;
}

__global__ void get_rowIdxbyWarp(int *rowPtrbyWarp, int *rowIdxbyWarp, int blc_row)
{
    int rowid = threadIdx.x + blockIdx.x * blockDim.x;
    if (rowid >= blc_row)
        return;

    int offset = rowPtrbyWarp[rowid];
    int stride = rowPtrbyWarp[rowid + 1] - rowPtrbyWarp[rowid];

    for (int i = offset; i < (offset + stride); i++)
    {
        rowIdxbyWarp[i] = rowid;
    }
}
__global__ void getStand(MAT_PTR_TYPE *rowptr, double *sum, double avg_len, int N)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ MAT_VAL_TYPE partialSum[256];

    if (idx < N)
    {
        // partialSum[threadIdx.x] = a[idx] * b[idx];
        partialSum[threadIdx.x] = pow(rowptr[idx + 1] - rowptr[idx] - avg_len, 2);
    }
    else
    {
        partialSum[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0)
    {
        if (threadIdx.x < i)
        {
            partialSum[threadIdx.x] += partialSum[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&sum[0], partialSum[0]);
    }
}

__global__ void beta_vecY(MAT_VAL_TYPE *d_y, MAT_VAL_TYPE beta, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_y[rowid] *= beta;
}

#define MASK_SIZE 256

__global__ void csr2bsr_get_ptr(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_PTR_TYPE *d_bsrptr,
                                int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;

    __shared__ unsigned int mask[MASK_SIZE];

    int rowid = bid * 4 + warpid;
    // if (rowid >= row) return;

    int start = d_csrptr[rowid >= row ? row : rowid] + laneid;
    int end = d_csrptr[(rowid + 1) >= row ? row : (rowid + 1)];

    int sum = 0;

    for (int i = 0; i < col; i += MASK_SIZE * 4 * 32)
    {
        int cur_end = (i + MASK_SIZE * 4 * 32) < col ? (i + MASK_SIZE * 4 * 32) : col;
        for (int id = tid; id < MASK_SIZE; id += blockDim.x)
        {
            mask[id] = 0;
        }
        __syncthreads();

        for (; start < end; start += WARP_SIZE)
        {
            int cid = d_csridx[start];
            if (cid < cur_end)
            {
                int key = (cid - i) / BSR_N;
                atomicOr(&(mask[key >> 5]), 1 << (key & 31));
            }
            else
            {
                break;
            }
        }
        __syncthreads();

        for (int id = tid; id < MASK_SIZE; id += blockDim.x)
        {
            unsigned int cur_num = mask[id];
            sum += __popc(cur_num);
        }
        __syncthreads();
    }

    sum = sum_warp_shfl_int(sum);
    __syncthreads();

    if (laneid == 0)
    {
        atomicAdd(&d_bsrptr[bid], sum);
    }
}

#define CONVERT_BIN 7

__global__ void csr2bsr_compute_bin(MAT_PTR_TYPE *d_bsrptr, int brow, int *bin_offset)
{
    int rid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rid >= brow)
        return;

    int len = d_bsrptr[rid + 1] - d_bsrptr[rid];

    if (len < 128)
    {
        atomicAdd(&bin_offset[0], 1);
    }
    else if (len >= 128 && len < 256)
    {
        atomicAdd(&bin_offset[1], 1);
    }
    else if (len >= 256 && len < 512)
    {
        atomicAdd(&bin_offset[2], 1);
    }
    else if (len >= 512 && len < 1024)
    {
        atomicAdd(&bin_offset[3], 1);
    }
    else if (len >= 1024 && len < 2048)
    {
        atomicAdd(&bin_offset[4], 1);
    }
    else if (len >= 2048 && len < 4096)
    {
        atomicAdd(&bin_offset[5], 1);
    }
    else
    {
        atomicAdd(&bin_offset[6], 1);
    }
    __syncthreads();
}

__global__ void csr2bsr_set_bin(MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *bin_rowidx, int *bin_offset, int *bin_size, int *max_num, int brow)
{
    int rid = blockIdx.x * blockDim.x + threadIdx.x;
    if (rid >= brow)
        return;

    int cur_Cub = d_bsrptr[rid + 1] - d_bsrptr[rid];
    int idx = 0;

    if (cur_Cub < 128)
    {
        idx = atomicAdd(&bin_size[0], 1);
        bin_rowidx[bin_offset[0] + idx] = rid;
    }
    else if (cur_Cub >= 128 && cur_Cub < 256)
    {
        idx = atomicAdd(&bin_size[1], 1);
        bin_rowidx[bin_offset[1] + idx] = rid;
    }
    else if (cur_Cub >= 256 && cur_Cub < 512)
    {
        idx = atomicAdd(&bin_size[2], 1);
        bin_rowidx[bin_offset[2] + idx] = rid;
    }
    else if (cur_Cub >= 512 && cur_Cub < 1024)
    {
        idx = atomicAdd(&bin_size[3], 1);
        bin_rowidx[bin_offset[3] + idx] = rid;
    }
    else if (cur_Cub >= 1024 && cur_Cub < 2048)
    {
        idx = atomicAdd(&bin_size[4], 1);
        bin_rowidx[bin_offset[4] + idx] = rid;
    }
    else if (cur_Cub >= 2048 && cur_Cub < 4096)
    {
        idx = atomicAdd(&bin_size[5], 1);
        bin_rowidx[bin_offset[5] + idx] = rid;
    }
    else
    {
        idx = atomicAdd(&bin_size[6], 1);
        bin_rowidx[bin_offset[6] + idx] = rid;
        atomicMax(max_num, cur_Cub);
    }
}

template <int SM_SIZE>
__global__ void csr2bsr_getidx(int *bin_rowidx, int *bin_offset, int bin,
                               MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                               MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                               int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    __shared__ int hashtable[SM_SIZE];
    __shared__ unsigned int maptable[SM_SIZE];
    __shared__ int nz_num[1];

    if (tid == 0)
    {
        nz_num[0] = 0;
    }

    for (int i = tid; i < SM_SIZE; i += blockDim.x)
    {
        hashtable[i] = -1;
    }

    for (int i = tid; i < SM_SIZE; i += blockDim.x)
    {
        maptable[i] = 0;
    }
    __syncthreads();

    int rowid = bin_rowidx[bin_row_offset];

    int start = (rowid * 4 + warpid) < row ? d_csrptr[rowid * 4 + warpid] : d_csrptr[row];
    int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

    for (int j = start + laneid; j < end; j += WARP_SIZE)
    {
        int cid = d_csridx[j];
        int key = cid / BSR_N;
        int hashadr = key & (SM_SIZE - 1);
        while (1)
        {
            int keyexist = hashtable[hashadr];
            if (keyexist == key)
            {
                atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                break;
            }
            else if (keyexist == -1)
            {
                int idx = atomicCAS(hashtable + hashadr, -1, key);
                if (idx == -1)
                {
                    atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                    break;
                }
            }
            else
            {
                hashadr = (hashadr + 1) & (SM_SIZE - 1);
            }
        }
    }
    __syncthreads();

    if (tid < WARP_SIZE)
    {
        for (int i = tid; i < SM_SIZE; i += WARP_SIZE)
        {
            unsigned int res_map = maptable[i];
            int res = hashtable[i];
            if (res != -1)
            {
                int ind = atomicAdd(&nz_num[0], 1);
                hashtable[ind] = res;
                maptable[ind] = res_map;
            }
        }
    }
    __syncthreads();

    int len = nz_num[0];

    int offset = d_bsrptr[rowid];
    int target, count;
    unsigned int target_map;
    unsigned short set_num = 0x0000ffff;
    for (int i = tid; i < len; i += blockDim.x)
    {
        target = hashtable[i];
        target_map = maptable[i];
        count = 0;

        for (int j = 0; j < len; j++)
        {
            count += ((unsigned int)(hashtable[j] - target) >> 31);
        }
        d_bsridx[offset + count] = target;
        d_bsrmap[offset + count] = target_map & set_num;
    }
    __syncthreads();

    MAT_VAL_TYPE *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
    for (int j = start + laneid; j < end; j += WARP_SIZE)
    {
        MAT_IDX_TYPE cid = d_csridx[j];
        MAT_VAL_TYPE val = d_csrval[j];
        int bcid = cid / BSR_N;

        int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
        int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
        cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
    }
    __syncthreads();
}

__global__ void csr2bsr_getidx_large(int *bin_rowidx, int *bin_offset, int bin,
                                     MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                                     MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                                     int brow, int bcol, int row, int col)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int laneid = tid & (WARP_SIZE - 1);
    int warpid = tid / WARP_SIZE;
    int bin_row_offset = bin_offset[bin] + bid;
    if (bin_row_offset >= bin_offset[bin + 1])
        return;

    __shared__ int hashtable[4096];
    __shared__ unsigned int maptable[4096];
    __shared__ int nz_num[1];

    int sum_len = 0;

    int rowid = bin_rowidx[bin_row_offset];

    int start1 = (rowid * 4 + warpid) < row ? (d_csrptr[rowid * 4 + warpid] + laneid) : d_csrptr[row];
    int start2 = start1;
    int end = (rowid * 4 + warpid + 1) < row ? d_csrptr[rowid * 4 + warpid + 1] : d_csrptr[row];

    for (int i = 0; i < col; i += 4096 * 4)
    {
        int cur_end = (i + 4096 * 4) < col ? (i + 4096 * 4) : col;

        if (tid == 0)
        {
            nz_num[0] = 0;
        }

        for (int id = tid; id < 4096; id += blockDim.x)
        {
            hashtable[id] = -1;
        }

        for (int id = tid; id < 4096; id += blockDim.x)
        {
            maptable[id] = 0;
        }
        __syncthreads();

        for (; start1 < end; start1 += WARP_SIZE)
        {
            int cid = d_csridx[start1];
            if (cid < cur_end)
            {
                int key = cid / BSR_N;
                int hashadr = key & (4096 - 1);
                while (1)
                {
                    int keyexist = hashtable[hashadr];
                    if (keyexist == key)
                    {
                        atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                        break;
                    }
                    else if (keyexist == -1)
                    {
                        int idx = atomicCAS(hashtable + hashadr, -1, key);
                        if (idx == -1)
                        {
                            atomicOr(maptable + hashadr, 1 << (warpid * 4 + (cid % 4)));
                            break;
                        }
                    }
                    else
                    {
                        hashadr = (hashadr + 1) & (4096 - 1);
                    }
                }
            }
            else
            {
                break;
            }
        }
        __syncthreads();

        if (tid < WARP_SIZE)
        {
            for (int id = tid; id < 4096; id += WARP_SIZE)
            {
                unsigned int res_map = maptable[id];
                int res = hashtable[id];
                if (res != -1)
                {
                    int ind = atomicAdd(&nz_num[0], 1);
                    hashtable[ind] = res;
                    maptable[ind] = res_map;
                }
            }
        }
        __syncthreads();

        int len = nz_num[0];

        int offset = d_bsrptr[rowid] + sum_len;
        int target, count;
        unsigned int target_map;
        unsigned short set_num = 0x0000ffff;
        for (int id = tid; id < len; id += blockDim.x)
        {
            target = hashtable[id];
            target_map = maptable[id];
            count = 0;

            for (int j = 0; j < len; j++)
            {
                count += ((unsigned int)(hashtable[j] - target) >> 31);
            }
            d_bsridx[offset + count] = target;
            d_bsrmap[offset + count] = target_map & set_num;
        }
        __syncthreads();

        MAT_VAL_TYPE *cur_bsrval = d_bsrval + (offset * (BSR_M * BSR_N));
        for (; start2 < end; start2 += WARP_SIZE)
        {

            MAT_IDX_TYPE cid = d_csridx[start2];
            if (cid < cur_end)
            {
                MAT_VAL_TYPE val = d_csrval[start2];
                int bcid = cid / BSR_N;

                int offset_cid = BinarySearch2(d_bsridx + offset, 0, len, bcid);
                int offset_idx = (warpid * BSR_M) + (cid % BSR_N);
                cur_bsrval[(offset_cid * (BSR_M * BSR_N)) + offset_idx] = val;
            }
            else
            {
                break;
            }
        }

        sum_len += len;
        __syncthreads();
    }
}

void CSR2BSR_step1(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_PTR_TYPE *d_bsrptr,
                   int brow, int bcol, int row, int col)
{
    int ThreadNum = 4 * WARP_SIZE;
    int BlockNum = brow;
    csr2bsr_get_ptr<<<BlockNum, ThreadNum>>>(d_csrptr, d_csridx, d_bsrptr, brow, bcol, row, col);
    cudaDeviceSynchronize();
}

void CSR2BSR_step2(MAT_PTR_TYPE *d_bsrptr, int brow)
{
    thrust::exclusive_scan(thrust::device, d_bsrptr, d_bsrptr + (brow + 1), d_bsrptr, 0);
    cudaDeviceSynchronize();
}

void CSR2BSR_step3(MAT_PTR_TYPE *d_csrptr, MAT_IDX_TYPE *d_csridx, MAT_VAL_TYPE *d_csrval,
                   MAT_PTR_TYPE *d_bsrptr, MAT_IDX_TYPE *d_bsridx, MAT_VAL_TYPE *d_bsrval, MAT_MAP_TYPE *d_bsrmap,
                   int brow, int bcol, int nnb, int row, int col)
{
    int *bin_offset, *bin_size;
    cudaMalloc((void **)&bin_offset, sizeof(int) * (CONVERT_BIN + 1));
    cudaMalloc((void **)&bin_size, sizeof(int) * CONVERT_BIN);
    cudaMemset(bin_offset, 0, sizeof(int) * (CONVERT_BIN + 1));
    cudaMemset(bin_size, 0, sizeof(int) * CONVERT_BIN);

    MAT_IDX_TYPE *bin_rowidx;
    cudaMalloc((void **)&bin_rowidx, sizeof(MAT_IDX_TYPE) * brow);
    int *max_num;
    cudaMalloc((void **)&max_num, sizeof(int));

    int ThreadNum = WARP_SIZE * 4;
    int BlockNum = (brow + ThreadNum - 1) / ThreadNum;

    csr2bsr_compute_bin<<<BlockNum, ThreadNum>>>(d_bsrptr, brow, bin_offset);
    cudaDeviceSynchronize();

    thrust::exclusive_scan(thrust::device, bin_offset, bin_offset + (CONVERT_BIN + 1), bin_offset, 0);

    csr2bsr_set_bin<<<BlockNum, ThreadNum>>>(d_bsrptr, bin_rowidx, bin_offset, bin_size, max_num, brow);
    cudaDeviceSynchronize();

    int max_len;
    cudaMemcpy(&max_len, max_num, sizeof(int), cudaMemcpyDeviceToHost);
    int *offset = (int *)malloc(sizeof(int) * (CONVERT_BIN + 1));
    cudaMemcpy(offset, bin_offset, sizeof(int) * (CONVERT_BIN + 1), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = CONVERT_BIN - 1; i >= 0; i--)
    {
        int row_num = offset[i + 1] - offset[i];
        ThreadNum = WARP_SIZE * 4;
        BlockNum = row_num;

        if (row_num)
        {
            switch (i)
            {
            case 0:
                csr2bsr_getidx<128><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 1:
                csr2bsr_getidx<256><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 2:
                csr2bsr_getidx<512><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                             d_csrptr, d_csridx, d_csrval,
                                                             d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                             brow, bcol, row, col);
                break;
            case 3:
                csr2bsr_getidx<1024><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 4:
                csr2bsr_getidx<2048><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 5:
                csr2bsr_getidx<4096><<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            case 6:
            {
                csr2bsr_getidx_large<<<BlockNum, ThreadNum>>>(bin_rowidx, bin_offset, i,
                                                              d_csrptr, d_csridx, d_csrval,
                                                              d_bsrptr, d_bsridx, d_bsrval, d_bsrmap,
                                                              brow, bcol, row, col);
                break;
            }
            }
            cudaDeviceSynchronize();
        }
    }
}

void CSR2BSR_GPU(hypre_CSRMatrix *A)
{
    if (!hypre_BSRTAG(A))
    {

        struct timeval t_start, t_end;
        hypre_BSRTAG(A) = 1;
        (hypre_BSR(A)) = (bsrMAT *)malloc(sizeof(bsrMAT));
        bsrMAT *bsrmat = (hypre_BSR(A));
        int *d_csrptr = hypre_CSRMatrixI(A);
        int *d_csridx = hypre_CSRMatrixJ(A);
        MAT_VAL_TYPE *d_csrval = hypre_CSRMatrixData(A);
        bsrmat->row = hypre_CSRMatrixNumRows(A);
        bsrmat->col = hypre_CSRMatrixNumCols(A);
        bsrmat->blc_row = (bsrmat->row + BSR_M - 1) / BSR_M;
        bsrmat->blc_col = (bsrmat->col + BSR_N - 1) / BSR_N;

        // csr2bsr step 1: get the block number of each block-row
        gettimeofday(&t_start, NULL);
        cudaMalloc((void **)&(bsrmat->blcPtr), sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->blcPtr, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        CSR2BSR_step1(d_csrptr, d_csridx, bsrmat->blcPtr, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col);
        gettimeofday(&t_end, NULL);
        csr2bsr_step1 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        // csr2bsr step1 over

        // csr2bsr step 2: pre-sum, get the bsrPtr array
        gettimeofday(&t_start, NULL);
        CSR2BSR_step2(bsrmat->blcPtr, bsrmat->blc_row);
        gettimeofday(&t_end, NULL);
        csr2bsr_step2 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        // csr2bsr step2 over

        cudaMemcpy(&(bsrmat->blc_num), &bsrmat->blcPtr[bsrmat->blc_row], sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);
        bsrmat->nnz = bsrmat->blc_num * BSR_M * BSR_N;
        bsrmat->avg_nnz = (double)hypre_CSRMatrixNumNonzeros(A) / (double)(bsrmat->blc_num);

        HYPRE_Real *result_gpu;
        cudaMalloc((void **)&result_gpu, sizeof(HYPRE_Real));
        cudaMemset(result_gpu, 0.0, sizeof(HYPRE_Real));
        int thread_num_stand = 256;
        int block_num_stand = (bsrmat->blc_row + thread_num_stand - 1) / thread_num_stand;
        double avg_len = (double)bsrmat->blc_num / (double)bsrmat->blc_row;

        getStand<<<block_num_stand, thread_num_stand>>>(bsrmat->blcPtr, result_gpu, avg_len, bsrmat->blc_row);
        cudaDeviceSynchronize();
        cudaMemcpy(&bsrmat->stand, result_gpu, sizeof(HYPRE_Real), cudaMemcpyDeviceToHost);

        bsrmat->stand = sqrtf(bsrmat->stand / bsrmat->blc_row);
        
        // csr2bsr step 3: get the blcIdx, blcVal, blcMap
        gettimeofday(&t_start, NULL);
        cudaMalloc((void **)&bsrmat->blcIdx, sizeof(MAT_IDX_TYPE) * bsrmat->blc_num);
        cudaMalloc((void **)&bsrmat->blcVal, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMalloc((void **)&bsrmat->blcMap, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        cudaMemset(bsrmat->blcVal, 0, sizeof(MAT_VAL_TYPE) * bsrmat->nnz);
        cudaMemset(bsrmat->blcMap, 0, sizeof(MAT_MAP_TYPE) * (bsrmat->blc_num + 1));

        CSR2BSR_step3(d_csrptr, d_csridx, d_csrval,
                      bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, bsrmat->blcMap,
                      bsrmat->blc_row, bsrmat->blc_col, bsrmat->blc_num, bsrmat->row, bsrmat->col);
        gettimeofday(&t_end, NULL);
        csr2bsr_step3 += (t_end.tv_sec - t_start.tv_sec) * 1000.0 + (t_end.tv_usec - t_start.tv_usec) / 1000.0;
        // csr2bsr step3 over
    }
}

void BSR_BALANCED_PREPROCESS_GPU(hypre_CSRMatrix *A)
{
#ifdef ADAPTIVE_AMGT_SPMV
    if (!hypre_BSRBALANCEDTAG(A) && hypre_BSR(A)->stand >= 12)
#else
    if (!hypre_BSRBALANCEDTAG(A))
#endif
    {
        hypre_BSRBALANCEDTAG(A) = 1;
        bsrMAT *bsrmat = (hypre_BSR(A));
        // load balanced preprocess

        cudaMalloc((void **)&bsrmat->rowPtrbyWarp, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));
        cudaMemset(bsrmat->rowPtrbyWarp, 0, sizeof(MAT_PTR_TYPE) * (bsrmat->blc_row + 1));

        int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum = (bsrmat->blc_row + ThreadNum - 1) / ThreadNum;

        get_rowPtrbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->rowPtrbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();

        thrust::exclusive_scan(thrust::device, bsrmat->rowPtrbyWarp, bsrmat->rowPtrbyWarp + (bsrmat->blc_row + 1), bsrmat->rowPtrbyWarp, 0);
        cudaDeviceSynchronize();

        cudaMemcpy(&bsrmat->warpnum, (bsrmat->rowPtrbyWarp) + bsrmat->blc_row, sizeof(MAT_PTR_TYPE), cudaMemcpyDeviceToHost);

        cudaMalloc((void **)&bsrmat->rowIdxbyWarp, sizeof(int) * bsrmat->warpnum);

        get_rowIdxbyWarp<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->blc_row);
        cudaDeviceSynchronize();
    }
}
void spmv_amgT_fp64(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    CSR2BSR_GPU(A);
    BSR_BALANCED_PREPROCESS_GPU(A);
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    gettimeofday1(&t1, NULL);
    bsrMAT *bsrmat = (hypre_BSR(A));
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;

#ifdef ADAPTIVE_AMGT_SPMV
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp64<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp64<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else

    bsr_spmv_balanced_tc_fp64<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal, dvecX, dvecY, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
    cudaDeviceSynchronize();
#endif
    gettimeofday1(&t2, NULL);
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}
__global__ void bsr_spmv_balanced_tc_fp32(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, float *d_blcVal,
                                          float *d_x, float *d_y, int blc_row, int blc_col, int row, int col,
                                          float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    uint32_t fragA[2] = {0}, fragB[2] = {0};
    float fragC[4] = {0};

    for (int i = start; i < end; i += 2)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragA[0]) : "f"((i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid]));

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"(d_x[xid + laneid_mod_4]));

        mma_m16n8k4_tf32_spmv(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[0] * alpha);
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            atomicAdd(&d_y[rowid], fragC[1] * alpha);
    }
}
__global__ void bsr_spmv_balanced_cc_fp32(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, float *d_blcVal,
                                          float *d_x, float *d_y, int blc_row, int blc_col, int row, int col,
                                          float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    float res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp32(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, float *d_blcVal,
                                 float *d_x, float *d_y,
                                 int blc_row, int blc_col, int row, int col, float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    uint32_t fragA[2] = {0}, fragB[2] = {0};
    float fragC[4] = {0};

    for (int i = start; i < end; i += 2)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragA[0]) : "f"((i + 1 >= end && laneid >= 16) ? 0 : cur_val[laneid]));

        int laneid_mod_4 = laneid & 3;
        int xid = laneid < 16 ? (d_blcCid[i] * BSR_N) : ((i + 1) < end ? d_blcCid[i + 1] * BSR_N : d_blcCid[i] * BSR_N);
        asm volatile("cvt.rna.tf32.f32  %0, %1;\n" : "=r"(fragB[0]) : "f"(d_x[xid + laneid_mod_4]));

        mma_m16n8k4_tf32_spmv(fragC, fragA, fragB);
    }

    fragC[0] += __shfl_down_sync(0xffffffff, fragC[0], 18, 32);
    fragC[1] += __shfl_down_sync(0xffffffff, fragC[1], 18, 32);

    if (laneid == 0)
    {
        int rowid = blc_rid * 4;
        if (rowid < row)
            d_y[rowid] = fragC[0] * alpha;
    }
    if (laneid == 4)
    {
        int rowid = blc_rid * 4 + 1;
        if (rowid < row)
            d_y[rowid] = fragC[1] * alpha;
    }
    if (laneid == 9)
    {
        int rowid = blc_rid * 4 + 2;
        if (rowid < row)
            d_y[rowid] = fragC[0] * alpha;
    }
    if (laneid == 13)
    {
        int rowid = blc_rid * 4 + 3;
        if (rowid < row)
            d_y[rowid] = fragC[1] * alpha;
    }
}
__global__ void bsr_spmv_cc_fp32(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, float *d_blcVal,
                                 float *d_x, float *d_y,
                                 int blc_row, int blc_col, int row, int col, float alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    float res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        float *cur_val = d_blcVal + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y[blc_rid * BSR_M + laneid] = alpha * res;
    }
}
__global__ void vec_64_to_32(MAT_VAL_TYPE *d_x_csr, float *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_x_bsr[rowid] = d_x_csr[rowid];
}

__global__ void vec_64_to_16(MAT_VAL_TYPE *d_x_csr, uint32_t *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    half *d_x_half = reinterpret_cast<half *>(&d_x_bsr[0]);
    d_x_half[rowid] = d_x_csr[rowid];
}

__global__ void vec_16_to_64(MAT_VAL_TYPE *d_x_csr, uint32_t *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    half *d_x_half = reinterpret_cast<half *>(&d_x_bsr[0]);
    d_x_csr[rowid] = d_x_half[rowid];
}

__global__ void vec_32_to_64(MAT_VAL_TYPE *d_x_csr, float *d_x_bsr, int row)
{
    int rowid = threadIdx.x + blockDim.x * blockIdx.x;
    if (rowid >= row)
        return;
    d_x_csr[rowid] = d_x_bsr[rowid];
}
__global__ void bsr_spmv_balanced_tc_fp16(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, uint32_t *d_blcVal,
                                          uint32_t *d_x, uint32_t *d_y, int blc_row, int blc_col, int row, int col,
                                          MAT_VAL_TYPE alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);
    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    half res, fragC[8] = {0};
    uint32_t fragA[2], fragB[2];

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;

    for (int i = start; i < end; i += 8)
    {
        int cur_blockid = laneid / 4 + i;
        uint32_t *cur_val = d_blcVal + (i * BSR_NNZ / 2);
        fragA[0] = cur_blockid < end ? cur_val[laneid * 2] : 0;
        fragA[1] = cur_blockid < end ? cur_val[laneid * 2 + 1] : 0;

        int xid = cur_blockid < end ? (d_blcCid[cur_blockid] * BSR_N / 2) : (d_blcCid[i] * BSR_N / 2);
        fragB[0] = d_x[xid];
        fragB[1] = d_x[xid + 1];

        mma_m8n8k4_fp16(fragC, fragA, fragB);
    }
    res = fragC[target_idx];

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        int rowid = blc_rid * 4 + laneid;
        if (rowid < row)
            atomicAdd(&d_y_half[rowid], res * (half)alpha);
    }
}
__global__ void bsr_spmv_cc_fp16(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, uint32_t *d_blcVal,
                                 uint32_t *d_x, uint32_t *d_y,
                                 int blc_row, int blc_col, int row, int col, half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    half *d_A_half = reinterpret_cast<half *>(&d_blcVal[0]);
    half *d_x_half = reinterpret_cast<half *>(&d_x[0]);
    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    half res = 0;
    for (int i = start + groupid; i < end; i += 8)
    {
        half *cur_val = d_A_half + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x_half[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        d_y_half[blc_rid * BSR_M + laneid] = alpha * res;
    }
}
__global__ void bsr_spmv_balanced_cc_fp16(int *rowPtrbyWarp, int *rowIdxbyWarp, int warp_num,
                                          MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, MAT_MAP_TYPE *d_blcMap, uint32_t *d_blcVal,
                                          uint32_t *d_x, uint32_t *d_y, int blc_row, int blc_col, int row, int col,
                                          half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);
    int groupid = laneid >> 2;
    int tid_in_group = laneid & 3;

    if (warpid >= warp_num)
        return;
    int blc_rid = rowIdxbyWarp[warpid];

    int start = d_blcPtr[blc_rid] + (warpid - rowPtrbyWarp[blc_rid]) * WARP_CAPACITY;
    int end = start + WARP_CAPACITY < d_blcPtr[blc_rid + 1] ? start + WARP_CAPACITY : d_blcPtr[blc_rid + 1];

    half *d_A_half = reinterpret_cast<half *>(&d_blcVal[0]);
    half *d_x_half = reinterpret_cast<half *>(&d_x[0]);
    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    half res = 0;

    for (int i = start + groupid; i < end; i += 8)
    {
        half *cur_val = d_A_half + i * BSR_NNZ;
        MAT_MAP_TYPE mapA = d_blcMap[i];

        int offset_b = d_blcCid[i] * BSR_N;

        for (int c = 0; c < BSR_N; c++)
        {
            int idx = tid_in_group * BSR_N + c;

            if (getbit(mapA, idx))
            {
                res += cur_val[idx] * d_x_half[offset_b + c];
            }
        }
    }
    __syncwarp();

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        atomicAdd(&d_y_half[blc_rid * BSR_M + laneid], res * alpha);
    }
}
__global__ void bsr_spmv_tc_fp16(MAT_PTR_TYPE *d_blcPtr, MAT_IDX_TYPE *d_blcCid, uint32_t *d_blcVal,
                                 uint32_t *d_x, uint32_t *d_y,
                                 int blc_row, int blc_col, int row, int col, half alpha)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int warpid = tid >> 5;
    int laneid = tid & (WARP_SIZE - 1);

    int blc_rid = warpid;
    if (blc_rid >= blc_row)
        return;

    half *d_y_half = reinterpret_cast<half *>(&d_y[0]);

    int start = d_blcPtr[blc_rid];
    int end = d_blcPtr[blc_rid + 1];

    half res, fragC[8] = {0};
    uint32_t fragA[2], fragB[2];

    int target_idx = laneid < 16 ? (3 & laneid) : (3 & laneid) + 4;

    for (int i = start; i < end; i += 8)
    {
        int cur_blockid = laneid / 4 + i;
        uint32_t *cur_val = d_blcVal + (i * BSR_NNZ / 2);
        fragA[0] = cur_blockid < end ? cur_val[laneid * 2] : 0;
        fragA[1] = cur_blockid < end ? cur_val[laneid * 2 + 1] : 0;

        int xid = cur_blockid < end ? (d_blcCid[cur_blockid] * BSR_N / 2) : (d_blcCid[i] * BSR_N / 2);
        fragB[0] = d_x[xid];
        fragB[1] = d_x[xid + 1];

        mma_m8n8k4_fp16(fragC, fragA, fragB);
    }
    res = fragC[target_idx];

    res += __shfl_down_sync(0xffffffff, res, 16);
    res += __shfl_down_sync(0xffffffff, res, 8);
    res += __shfl_down_sync(0xffffffff, res, 4);

    if (laneid < 4)
    {
        int rowid = blc_rid * 4 + laneid;
        if (rowid < row)
            d_y_half[rowid] = res * (half)alpha;
    }
}
void spmv_amgT_fp16(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    CSR2BSR_GPU(A);
    BSR_BALANCED_PREPROCESS_GPU(A);
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    bsrMAT *bsrmat = (hypre_BSR(A));
    if (!hypre_VectorSpaceTag(A)) // 判断 A的辅助x和y空间是否已经存在
    {
        cudaMalloc((void **)&bsrmat->dVecX_fp16, sizeof(uint32_t) * ((bsrmat->col + 1) / 2));
        cudaMalloc((void **)&bsrmat->dVecY_fp16, sizeof(uint32_t) * ((bsrmat->row + 1) / 2));
        hypre_VectorSpaceTag(A) = 1;
    }

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    int BlockNum_x = (bsrmat->col + ThreadNum - 1) / ThreadNum;
    int BlockNum_y = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    vec_64_to_16<<<BlockNum_x, ThreadNum>>>(dvecX, bsrmat->dVecX_fp16, bsrmat->col); // 将double向量中的数赋值到float中

    cudaDeviceSynchronize();

    vec_64_to_16<<<BlockNum_y, ThreadNum>>>(dvecY, bsrmat->dVecY_fp16, bsrmat->row);
    cudaDeviceSynchronize();

    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;

    if (!hypre_MixedPrecisionTag(A) || hypre_BSR(A)->blcVal_fp16 == NULL) // 申请额外空间
    {
        int ThreadNum_val_convert = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum_spgemm_A = (bsrmat->nnz + ThreadNum_val_convert - 1) / ThreadNum_val_convert;
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&bsrmat->blcVal_fp16, sizeof(uint32_t) * ((bsrmat->nnz + 1) / 2));
        cudaMemset(bsrmat->blcVal_fp16, 0, sizeof(uint32_t) * ((bsrmat->nnz + 1) / 2));
        bsr_val_fp64_to_16<<<BlockNum_spgemm_A, ThreadNum_val_convert>>>(bsrmat->blcVal, bsrmat->blcVal_fp16, bsrmat->nnz);
        cudaDeviceSynchronize();
    }
    gettimeofday1(&t1, NULL);
#ifdef ADAPTIVE_AMGT_SPMV
    // printf("half!!!!!\n");
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp16<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp16<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else
    // check value
    bsr_spmv_balanced_tc_fp16<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp16, bsrmat->dVecX_fp16, bsrmat->dVecY_fp16, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
    cudaDeviceSynchronize();

#endif
    gettimeofday1(&t2, NULL);
    vec_16_to_64<<<BlockNum2, ThreadNum>>>(dvecY, bsrmat->dVecY_fp16, bsrmat->row);
    cudaDeviceSynchronize();

    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}

void spmv_amgT_fp32(HYPRE_Int trans,
                    HYPRE_Complex alpha,
                    hypre_CSRMatrix *A,
                    hypre_Vector *x,
                    HYPRE_Complex beta,
                    hypre_Vector *y,
                    HYPRE_Int offset)
{
    struct timeval t1, t2;
    gettimeofday1(&t1, NULL);
    CSR2BSR_GPU(A);
    BSR_BALANCED_PREPROCESS_GPU(A);
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    bsrMAT *bsrmat = (hypre_BSR(A));
    if (!hypre_VectorSpaceTag(A)) // 判断 A的辅助x和y空间是否已经存在
    {
        cudaMalloc((void **)&bsrmat->dVecX_fp32, sizeof(float) * bsrmat->col);
        cudaMalloc((void **)&bsrmat->dVecY_fp32, sizeof(float) * bsrmat->row);
        hypre_VectorSpaceTag(A) = 1;
    }

    int ThreadNum = WARP_SIZE * WARP_NUM_SPMV;
    int BlockNum_b = (bsrmat->warpnum + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum = (bsrmat->blc_row + WARP_NUM_SPMV - 1) / WARP_NUM_SPMV;
    int BlockNum2 = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    int BlockNum_x = (bsrmat->col + ThreadNum - 1) / ThreadNum;
    int BlockNum_y = (bsrmat->row + ThreadNum - 1) / ThreadNum;
    MAT_VAL_TYPE *dvecX = hypre_VectorData(x);
    MAT_VAL_TYPE *dvecY = hypre_VectorData(y);
    if (beta != 1)
    {
        beta_vecY<<<BlockNum2, ThreadNum>>>(dvecY, beta, bsrmat->row);
        cudaDeviceSynchronize();
    }
    vec_64_to_32<<<BlockNum_x, ThreadNum>>>(dvecX, bsrmat->dVecX_fp32, bsrmat->col); // 将double向量中的数赋值到float中

    cudaDeviceSynchronize();

    vec_64_to_32<<<BlockNum_y, ThreadNum>>>(dvecY, bsrmat->dVecY_fp32, bsrmat->row);
    cudaDeviceSynchronize();

    double stand = bsrmat->stand;
    double avgnz = bsrmat->avg_nnz;

    if (!hypre_MixedPrecisionTag(A) || hypre_BSR(A)->blcVal_fp32 == NULL) // 申请额外空间
    {
        // printf("no mixed!!!!!!\n");
        int ThreadNum_val_convert = WARP_SIZE * WARP_NUM_SPMV;
        int BlockNum_spgemm_A = (bsrmat->nnz + ThreadNum_val_convert - 1) / ThreadNum_val_convert;
        hypre_MixedPrecisionTag(A) = 1;
        cudaMalloc((void **)&bsrmat->blcVal_fp32, sizeof(float) * bsrmat->nnz);
        cudaMemset(bsrmat->blcVal_fp32, 0, sizeof(float) * bsrmat->nnz);
        bsr_val_fp64_to_32<<<BlockNum_spgemm_A, ThreadNum_val_convert>>>(bsrmat->blcVal, bsrmat->blcVal_fp32, bsrmat->nnz);
        cudaDeviceSynchronize();
    }
    gettimeofday1(&t1, NULL);
#ifdef ADAPTIVE_AMGT_SPMV
    if (stand >= 12 && avgnz >= 10)
    {
        // ===tensor core, balanced===
        bsr_spmv_balanced_tc_fp32<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand >= 12 && avgnz < 10)
    {
        // ===cuda core, balanced===
        bsr_spmv_balanced_cc_fp32<<<BlockNum_b, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else if (stand < 12 && avgnz >= 10)
    {
        // ===tensor core===
        bsr_spmv_tc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }
    else
    {
        // ===cuda core===
        bsr_spmv_cc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcMap, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, alpha);
        cudaDeviceSynchronize();
        // ===============================
    }

#else
    // check value
    bsr_spmv_balanced_tc_fp32<<<BlockNum, ThreadNum>>>(bsrmat->rowPtrbyWarp, bsrmat->rowIdxbyWarp, bsrmat->warpnum, bsrmat->blcPtr, bsrmat->blcIdx, bsrmat->blcVal_fp32, bsrmat->dVecX_fp32, bsrmat->dVecY_fp32, bsrmat->blc_row, bsrmat->blc_col, bsrmat->row, bsrmat->col, (float)alpha);
    cudaDeviceSynchronize();

#endif
    gettimeofday1(&t2, NULL);
    vec_32_to_64<<<BlockNum2, ThreadNum>>>(dvecY, bsrmat->dVecY_fp32, bsrmat->row);
    cudaDeviceSynchronize();
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", bsrmat->row);
    printf("spmv_kernel_n=%d\n", bsrmat->col);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;
}

HYPRE_Int
hypre_CSRMatrixMatvecCusparseNewAPI(HYPRE_Int trans,
                                    HYPRE_Complex alpha,
                                    hypre_CSRMatrix *A,
                                    hypre_Vector *x,
                                    HYPRE_Complex beta,
                                    hypre_Vector *y,
                                    HYPRE_Int offset)
{
    spmv_times++;

    struct timeval t1, t2;
#ifdef Hypre_AMGT

#ifdef MIXED_PRESION
    AMGT_PRECISION precision = get_calculationPrecison(hypre_Level(A));
    // spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);
    if (precision == AMGT_DOUBLE)
    {
        // printf("DOUBLE\n");
        spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);
    }
    else if (precision == AMGT_FLOAT)
    {
        // printf("%d\n", hypre_Level(A));
        // printf("FLOAT SpMV\n");
        spmv_amgT_fp32(trans, alpha, A, x, beta, y, offset);

        // printf("End SpMV\n");
    }
    else
    {
        // printf("half spmv\n");
        spmv_amgT_fp16(trans, alpha, A, x, beta, y, offset);
        // printf("half spmv end\n");
    }
#else

    spmv_amgT_fp64(trans, alpha, A, x, beta, y, offset);

#endif

    // printf("time_spmv:%lf\n", time_spmv_sum);
#else
    // #if 0
    gettimeofday1(&t1, NULL);
    HYPRE_Int num_vectors = hypre_VectorNumVectors(x);
    HYPRE_Int num_cols = trans ? hypre_CSRMatrixNumRows(A) : hypre_CSRMatrixNumCols(A);
    HYPRE_Int num_rows = trans ? hypre_CSRMatrixNumCols(A) : hypre_CSRMatrixNumRows(A);
    hypre_CSRMatrix *AT;
    hypre_CSRMatrix *B;
    /* SpMV data */
    size_t bufferSize = 0;
    char *dBuffer = hypre_CSRMatrixGPUMatSpMVBuffer(A);
    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    const cudaDataType data_type = hypre_HYPREComplexToCudaDataType();
    const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();

    /* Local cusparse descriptor variables */
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    cusparseDnMatDescr_t matX, matY;

    /* We handle the transpose explicitly to ensure the same output each run
     * and for potential performance improvement memory for AT */
    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &AT, 1);
        B = AT;
    }
    else
    {
        B = A;
    }

    /* Create cuSPARSE vector data structures */
    matA = hypre_CSRMatrixToCusparseSpMat(B, offset);
    if (num_vectors == 1)
    {
        // printf("matrix-vec\n");
        vecX = hypre_VectorToCusparseDnVec(x, 0, num_cols);
        vecY = hypre_VectorToCusparseDnVec(y, offset, num_rows - offset);
    }
    else
    {
        // printf("matrix-matrix\n");
        matX = hypre_VectorToCusparseDnMat(x);
        matY = hypre_VectorToCusparseDnMat(y);
    }

    if (!dBuffer)
    {
        // printf("!dBuffer\n");
        if (num_vectors == 1)
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        vecX,
                                                        &beta,
                                                        vecY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMV_ALG,
                                                        &bufferSize));
        }
        else
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMM_bufferSize(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        matX,
                                                        &beta,
                                                        matY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMM_ALG,
                                                        &bufferSize));
        }

        dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
        hypre_CSRMatrixGPUMatSpMVBuffer(A) = dBuffer;

#if CUSPARSE_VERSION >= CUSPARSE_NEWSPMM_VERSION
        if (num_vectors > 1)
        {
            HYPRE_CUSPARSE_CALL(cusparseSpMM_preprocess(handle,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                                                        &alpha,
                                                        matA,
                                                        matX,
                                                        &beta,
                                                        matY,
                                                        data_type,
                                                        HYPRE_CUSPARSE_SPMM_ALG,
                                                        dBuffer));
        }
#endif
    }
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;

    gettimeofday1(&t1, NULL);
    if (num_vectors == 1)
    {
        HYPRE_CUSPARSE_CALL(cusparseSpMV(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         vecX,
                                         &beta,
                                         vecY,
                                         data_type,
                                         HYPRE_CUSPARSE_SPMV_ALG,
                                         dBuffer));
        // cudaDeviceSynchronize();
    }
    else
    {
        HYPRE_CUSPARSE_CALL(cusparseSpMM(handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &alpha,
                                         matA,
                                         matX,
                                         &beta,
                                         matY,
                                         data_type,
                                         HYPRE_CUSPARSE_SPMM_ALG,
                                         dBuffer));
        //   cudaDeviceSynchronize();
    }
    gettimeofday1(&t2, NULL);
    double time_spmv_kernel_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
#ifdef PRINT_KERNEL_PERFORMANCE
    printf("spmv_kernel_m=%d\n", num_rows);
    printf("spmv_kernel_n=%d\n", num_cols);
    printf("spmv_kernel_time=%lf\n", time_spmv_kernel_time);
#endif
    time_spmv_sum += time_spmv_kernel_time;

    gettimeofday1(&t1, NULL);
#if defined(HYPRE_USING_GPU)
    hypre_SyncComputeStream(hypre_handle());
#endif

    /* Free memory */
    HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
    if (num_vectors == 1)
    {
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
    }
    else
    {
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnMat(matX));
        HYPRE_CUSPARSE_CALL(cusparseDestroyDnMat(matY));
    }
    if (trans)
    {
        hypre_CSRMatrixDestroy(AT);
    }
    gettimeofday1(&t2, NULL);
    time_spmv_preprocess += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //    gettimeofday1(&t2, NULL);
    //    time_spmv_sum += (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
    //    printf("time_spmv:%lf\n",time_spmv_sum);

#endif
    return hypre_error_flag;
}

#else // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparseOldAPI(HYPRE_Int trans,
                                    HYPRE_Complex alpha,
                                    hypre_CSRMatrix *A,
                                    hypre_Vector *x,
                                    HYPRE_Complex beta,
                                    hypre_Vector *y,
                                    HYPRE_Int offset)
{

    printf("==============old one \n");
#ifdef HYPRE_BIGINT
#error "ERROR: cusparse old API should not be used when bigint is enabled!"
#endif
    cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    cusparseMatDescr_t descr = hypre_CSRMatrixGPUMatDescr(A);
    hypre_CSRMatrix *B;

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &B, 1);
    }
    else
    {
        B = A;
    }

    HYPRE_CUSPARSE_CALL(hypre_cusparse_csrmv(handle,
                                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                                             hypre_CSRMatrixNumRows(B) - offset,
                                             hypre_CSRMatrixNumCols(B),
                                             hypre_CSRMatrixNumNonzeros(B),
                                             &alpha,
                                             descr,
                                             hypre_CSRMatrixData(B),
                                             hypre_CSRMatrixI(B) + offset,
                                             hypre_CSRMatrixJ(B),
                                             hypre_VectorData(x),
                                             &beta,
                                             hypre_VectorData(y) + offset));

    if (trans)
    {
        hypre_CSRMatrixDestroy(B);
    }

    return hypre_error_flag;
}

#endif // #if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION

HYPRE_Int
hypre_CSRMatrixMatvecCusparse(HYPRE_Int trans,
                              HYPRE_Complex alpha,
                              hypre_CSRMatrix *A,
                              hypre_Vector *x,
                              HYPRE_Complex beta,
                              hypre_Vector *y,
                              HYPRE_Int offset)
{
#if CUSPARSE_VERSION >= CUSPARSE_NEWAPI_VERSION
    /* Luke E: The generic API is techinically supported on 10.1,10.2 as a preview,
     * with Dscrmv being deprecated. However, there are limitations.
     * While in Cuda < 11, there are specific mentions of using csr2csc involving
     * transposed matrix products with dcsrm*,
     * they are not present in SpMV interface.
     */
    hypre_CSRMatrixMatvecCusparseNewAPI(trans, alpha, A, x, beta, y, offset);

#else
    hypre_CSRMatrixMatvecCusparseOldAPI(trans, alpha, A, x, beta, y, offset);
#endif

    return hypre_error_flag;
}

#endif // #if defined(HYPRE_USING_CUSPARSE)

#if defined(HYPRE_USING_ROCSPARSE)
HYPRE_Int
hypre_CSRMatrixMatvecRocsparse(HYPRE_Int trans,
                               HYPRE_Complex alpha,
                               hypre_CSRMatrix *A,
                               hypre_Vector *x,
                               HYPRE_Complex beta,
                               hypre_Vector *y,
                               HYPRE_Int offset)
{
    rocsparse_handle handle = hypre_HandleCusparseHandle(hypre_handle());
    rocsparse_mat_descr descr = hypre_CSRMatrixGPUMatDescr(A);
    rocsparse_mat_info info = hypre_CSRMatrixGPUMatInfo(A);

    hypre_CSRMatrix *B;

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &B, 1);
    }
    else
    {
        B = A;
    }

    HYPRE_ROCSPARSE_CALL(hypre_rocsparse_csrmv(handle,
                                               rocsparse_operation_none,
                                               hypre_CSRMatrixNumRows(B) - offset,
                                               hypre_CSRMatrixNumCols(B),
                                               hypre_CSRMatrixNumNonzeros(B),
                                               &alpha,
                                               descr,
                                               hypre_CSRMatrixData(B),
                                               hypre_CSRMatrixI(B) + offset,
                                               hypre_CSRMatrixJ(B),
                                               info,
                                               hypre_VectorData(x),
                                               &beta,
                                               hypre_VectorData(y) + offset));

    if (trans)
    {
        hypre_CSRMatrixDestroy(B);
    }

    return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#if defined(HYPRE_USING_ONEMKLSPARSE)
HYPRE_Int
hypre_CSRMatrixMatvecOnemklsparse(HYPRE_Int trans,
                                  HYPRE_Complex alpha,
                                  hypre_CSRMatrix *A,
                                  hypre_Vector *x,
                                  HYPRE_Complex beta,
                                  hypre_Vector *y,
                                  HYPRE_Int offset)
{
    sycl::queue *compute_queue = hypre_HandleComputeStream(hypre_handle());
    hypre_CSRMatrix *AT;
    oneapi::mkl::sparse::matrix_handle_t matA_handle = hypre_CSRMatrixGPUMatHandle(A);
    hypre_GPUMatDataSetCSRData(A);

    if (trans)
    {
        hypre_CSRMatrixTransposeDevice(A, &AT, 1);
        hypre_GPUMatDataSetCSRData(AT);
        matA_handle = hypre_CSRMatrixGPUMatHandle(AT);
    }

    HYPRE_ONEMKL_CALL(oneapi::mkl::sparse::gemv(*compute_queue,
                                                oneapi::mkl::transpose::nontrans,
                                                alpha,
                                                matA_handle,
                                                hypre_VectorData(x),
                                                beta,
                                                hypre_VectorData(y) + offset)
                          .wait());

    if (trans)
    {
        hypre_CSRMatrixDestroy(AT);
    }

    return hypre_error_flag;
}
#endif // #if defined(HYPRE_USING_ROCSPARSE)

#endif // #if defined(HYPRE_USING_GPU) || defined(HYPRE_USING_DEVICE_OPENMP)
