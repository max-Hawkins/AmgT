#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include "sub_files/mmio_highlevel.h"
#include "sub_files/my_solver.h"
#include "sub_files/subfunction.h"
#include <sys/stat.h>
#include <sys/types.h>

#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "seq_mv.h"
#include "ex.h"
#include "_hypre_parcsr_ls.h"
// #include "_hypre_utilities.hpp"
// #include "seq_mv/seq_mv.hpp"


#include <cublas_v2.h>
#include <cuda_runtime.h>
#include "nvml.h"
#include "nvmlClass.h"
#include <nvtx3/nvToolsExt.h>

#define HYPRE_CUSPARSE_SPMV_ALG CUSPARSE_SPMV_ALG_DEFAULT
// #define HYPRE_USING_CUSPARSE


// *************** FOR ERROR CHECKING *******************
#ifndef CUDA_RT_CALL
#define CUDA_RT_CALL( call )                                                                                           \
    {                                                                                                                  \
        auto status = static_cast<cudaError_t>( call );                                                                \
        if ( status != cudaSuccess )                                                                                   \
            fprintf( stderr,                                                                                           \
                     "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "                                        \
                     "with "                                                                                           \
                     "%s (%d).\n",                                                                                     \
                     #call,                                                                                            \
                     __LINE__,                                                                                         \
                     __FILE__,                                                                                         \
                     cudaGetErrorString( status ),                                                                     \
                     status );                                                                                         \
    }
#endif  // CUDA_RT_CALL

#define CHECK_NVML(result, message) \
    if (result != NVML_SUCCESS) { \
        printf("%s : %s", message, nvmlErrorString(result)); \
        return 1; \
    }

#ifdef HYPRE_EXVIS
#include "vis.c"
#endif

#define my_min(a, b) (((a) < (b)) ? (a) : (b))

typedef struct _temp_data
{
    int m;
    int n;
    int nnzA;
    int isSymmetricA;
    int num1;
    int vec1;
} _temp_data;

#define ReadMMFile 1
#define ReadHypreFile 0

void cublas_calculate(){
    cublasHandle_t handle;

    /* Initialize CUBLAS */
    CUDA_RT_CALL( cublasCreate( &handle ) );
}

int main(int argc, char **argv)
{
    cudaSetDevice(0);
    int m, n, nnzA, isSymmetricA;
    int *row_ptr; // the csr row pointer array of matrix A
    int *col_idx; // the csr column index array of matrix A
    int *cpu_row_ptr;
    int *cpu_col_idx;
    double *cpu_val;
    double *cpu_bval;

    long num_trials = 1000;
    char *kernel_type = "TC"; // Default to tensor cores
    int kernel_type_int = -1;

    if (argc >= 4) {
        num_trials = atoi(argv[2]);
        kernel_type = argv[3];
        if (strcmp(kernel_type, "TC") == 0){
            kernel_type_int = 0;
        }else if(strcmp(kernel_type, "CC") == 0){
            kernel_type_int = 1;
        }else if(strcmp(kernel_type, "cusparse") == 0){
            kernel_type_int = 2;
        }else{
            printf("Error: Kernel type must be either TC or CC or cusparse\n");
            return 1;
        }
    }else{
        printf("Error. Must pass in Number of trials and kernel type\n");
        printf("Usage: ./test_new <matrix_file> <num_trials> <kernel_type>\n");
        printf("\tkernel_type: TC (tensor cores), CC (CUDA cores), or cusparse\n");
        return 1;
    }
    char *filename_matrix = argv[1];
    // char *lastSlash = strrchr(filename_matrix, '/');
    // char *lastDot = strrchr(filename_matrix, '.');

    // if (lastSlash != NULL && lastDot != NULL && lastSlash < lastDot)
    // {
    //     // 计算截取的字符串长度
    //     size_t length = lastDot - (lastSlash + 1);
    // }

    // Extract matrix name and block sparsity from path
    char matrix_name[256];
    char block_sparsity[8];
    char *lastSlash = strrchr(filename_matrix, '/');
    char *lastDot = strrchr(filename_matrix, '.');
    char *lastUnderscore = strrchr(filename_matrix, '_');

    // Extract matrix name
    if (lastSlash != NULL && lastDot != NULL && lastSlash < lastDot) {
        size_t length = lastDot - (lastSlash + 1);
        strncpy(matrix_name, lastSlash + 1, length);
        matrix_name[length] = '\0';
    } else if (lastDot != NULL) {
        size_t length = lastDot - filename_matrix;
        strncpy(matrix_name, filename_matrix, length);
        matrix_name[length] = '\0';
    } else {
        strcpy(matrix_name, filename_matrix);
    }

    // Extract block sparsity number (assuming format: *_nnz_X.mtx where X is 1-16)
    if (lastUnderscore != NULL && lastDot != NULL && lastUnderscore < lastDot) {
        strncpy(block_sparsity, lastUnderscore + 1, lastDot - lastUnderscore - 1);
        block_sparsity[lastDot - lastUnderscore - 1] = '\0';
    } else {
        strcpy(block_sparsity, "unknown");
    }

    // Create unique directory name with matrix name, kernel type, and block sparsity
    char dir_path[512];
    snprintf(dir_path, sizeof(dir_path), "data/%s_%s", matrix_name, kernel_type);

    // Create directory if it doesn't exist, otherwise overwrite
    struct stat st = {0};
    if (stat(dir_path, &st) == -1) {
        mkdir(dir_path, 0700);
    }

    printf("Matrix name: %s\n", matrix_name);
    printf("Block sparsity: %s\n", block_sparsity);
    printf("Directory path: %s\n", dir_path);

    std::string const nvml_csv_filename = { std::string(dir_path) + "/gpuStats.csv" };

    int i;
    int myid, num_procs;
    int N;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Initialize HYPRE */
    HYPRE_Init();
    /* use vendor implementation for SpxGEMM */
    HYPRE_SetSpGemmUseVendor(1);
    HYPRE_SetMemoryLocation(HYPRE_MEMORY_DEVICE);
    /* setup AMG on GPUs */
    HYPRE_SetExecutionPolicy(HYPRE_EXEC_DEVICE);


    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);
    int num1;
    int vec1;
    _temp_data pack_data;
    MPI_Datatype newtype;
    MPI_Type_contiguous(6, MPI_INT, &newtype);
    MPI_Type_commit(&newtype);

    printf("Num procs: %d\n", num_procs);
    printf("Num trials: %d\n", num_trials);
    printf("MatrixFilename: %s\n", filename_matrix);

    unsigned int device_count;

    CHECK_NVML(nvmlInit(), "Failed to initialize NVML");
    CHECK_NVML(nvmlDeviceGetCount(&device_count), "Failed to get device count");

    int dev {};
    cudaGetDevice( &dev );
    CUDA_RT_CALL( cudaSetDevice( dev ) );
    CUDA_RT_CALL(cudaDeviceSynchronize());


    // Create NVML class to retrieve GPU stats
    nvmlClass nvml(dev, nvml_csv_filename);
    printf("Preparing for benchmark...\n");


    FILE *f;
    MM_typecode matcode;
    int ret_code;
    int num_rows, num_cols, nnz;
    int *I, *J;
    double *val;

    // Open the matrix market file
    f = fopen(filename_matrix, "r");
    if (f == NULL) {
        printf("Could not open matrix file\n");
        return -1;
    }

    // Read banner and size
    if (mm_read_banner(f, &matcode) != 0) {
        printf("Could not process Matrix Market banner\n");
        return -1;
    }

    if (mm_read_mtx_crd_size(f, &num_rows, &num_cols, &nnz) != 0) {
        printf("Could not read matrix size\n");
        return -1;
    }

    HYPRE_Int big_init = 0; // Don't do big init
    HYPRE_MemoryLocation test_A_mem_loc = HYPRE_MEMORY_HOST;

    hypre_CSRMatrix *test_A;
    test_A = hypre_CSRMatrixCreate(num_rows, num_cols, nnz);

    // Print matrix info for debugging
    printf("Matrix size: %d x %d with %d nonzeros\n", num_rows, num_cols, nnz);
    if (nnz <= 0 || num_rows <= 0 || num_cols <= 0) {
        printf("Invalid matrix dimensions or nonzeros\n");
        return -1;
    }

    int *A_i = (int *)malloc((num_rows + 1) * sizeof(int)); // Row offsets?
    int *A_j = (int *)malloc(nnz * sizeof(int)); // Column indices
    double *A_data = (double *)malloc(nnz * sizeof(double)); // Non-zero values

    int prev_val_row = 0;
    int cur_val_row = 0;
    A_i[0] = 0;

    // Read matrix entries
    for (i = 0; i < nnz; i++) {
        if (mm_is_pattern(matcode)) {
            if (fscanf(f, "%d %d\n", &cur_val_row, &A_j[i]) != 2) {
                printf("Error reading pattern entry %d\n", i);
                break;
            }
            A_data[i] = 1.0;
        } else {
            if (fscanf(f, "%d %d %lg\n", &cur_val_row, &A_j[i], &A_data[i]) != 3) {
                printf("Error reading value entry %d\n", i);
                break;
            }
        }
        // Adjust for 0-based indexing
        // cur_val_row--;
        A_j[i]--;

        if (prev_val_row != cur_val_row) {
            // printf("cur_val_row=%d, i=%d\n", cur_val_row, i);
            for(int z = prev_val_row; z <= cur_val_row; z++)
                A_i[z] = i;
        }
        prev_val_row = cur_val_row;
    }
    // printf("I=%d, A_j[%d]=%d, A_data[%d]=%lg\n", cur_val_row, i, A_j[i], i, A_data[i]);
    A_i[num_rows] = nnz;

    fclose(f);

    // Set Hypre CSR Matrix
    hypre_CSRMatrixI(test_A) = (HYPRE_Int *)A_i;
    hypre_CSRMatrixJ(test_A) = (HYPRE_Int *)A_j;
    hypre_CSRMatrixData(test_A) = (HYPRE_Complex *)A_data;

    HYPRE_Int num_nonzeros = hypre_CSRMatrixNumNonzeros(test_A);
    printf("Num nonzeros: %d\n", num_nonzeros);

    // Print A_i, A_j, A_data
    // for (i = 0; i < num_rows + 1; i++) {
    //     printf("A_i[%d]=%d\n", i, A_i[i]);
    // }
    // for (i = 0; i < num_nonzeros; i++) {
    //     printf("A_j[%d]=%d, A_data[%d]=%lg\n", i, A_j[i], i, A_data[i]);
    // }
    // Test output of Matrix
    // hypre_CSRMatrixPrintMM(test_A, 1,1, 0, "hypre_csr_test_A.mtx");

    // Move data to device
    HYPRE_Int *A_i_gpu = (HYPRE_Int *)gpu_malloc(sizeof(HYPRE_Int) * (num_rows + 1));
    cudaMemcpy(A_i_gpu, A_i, sizeof(HYPRE_Int) * (num_rows + 1), cudaMemcpyHostToDevice);
    hypre_CSRMatrixI(test_A) = A_i_gpu;
    HYPRE_Int *A_j_gpu = (HYPRE_Int *)gpu_malloc(sizeof(HYPRE_Int) * nnz);
    cudaMemcpy(A_j_gpu, A_j, sizeof(HYPRE_Int) * nnz, cudaMemcpyHostToDevice);
    hypre_CSRMatrixJ(test_A) = A_j_gpu;
    HYPRE_Complex *A_data_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex) * nnz);
    cudaMemcpy(A_data_gpu, A_data, sizeof(HYPRE_Complex) * nnz, cudaMemcpyHostToDevice);
    hypre_CSRMatrixData(test_A) = A_data_gpu;



    //
    // Create the dense vectors
    //
    HYPRE_Int vector_size = num_cols;

    hypre_Vector *x = hypre_SeqVectorCreate(vector_size);
    hypre_Vector *y = hypre_SeqVectorCreate(vector_size);

    // Set memory type to device
    hypre_VectorMemoryLocation(x) = HYPRE_MEMORY_DEVICE;
    hypre_VectorMemoryLocation(y) = HYPRE_MEMORY_DEVICE;

    // Initialize the vectors (this will handle the GPU memory allocation)
    hypre_SeqVectorInitialize(x);
    hypre_SeqVectorInitialize(y);
    HYPRE_Complex *x_data = (HYPRE_Complex *)malloc(sizeof(HYPRE_Complex) * vector_size);
    HYPRE_Complex *y_data = (HYPRE_Complex *)malloc(sizeof(HYPRE_Complex) * vector_size);
    for(int i = 0; i < vector_size; i++){
        x_data[i] = (HYPRE_Complex)1.0;
        y_data[i] = (HYPRE_Complex)0.0;
    }
    HYPRE_Complex *x_data_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex) * vector_size);
    HYPRE_Complex *y_data_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex) * vector_size);
    // Transfer data to GPU
    cudaMemcpy(x_data_gpu, x_data, sizeof(HYPRE_Complex) * vector_size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_data_gpu, y_data, sizeof(HYPRE_Complex) * vector_size, cudaMemcpyHostToDevice);
    hypre_VectorData(x) = x_data_gpu;
    hypre_VectorData(y) = y_data_gpu;
    hypre_SeqVectorInitialize(x);
    hypre_SeqVectorInitialize(y);

    // Prepare for SpMV kernel
    HYPRE_Int trans = 0;
    HYPRE_Int *trans_gpu = (HYPRE_Int *)gpu_malloc(sizeof(HYPRE_Int));
    cudaMemcpy(trans_gpu, &trans, sizeof(HYPRE_Int), cudaMemcpyHostToDevice);

    HYPRE_Complex alpha = 1.0;
    HYPRE_Complex *alpha_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex));
    cudaMemcpy(alpha_gpu, &alpha, sizeof(HYPRE_Complex), cudaMemcpyHostToDevice);
    HYPRE_Complex beta = 0.0;
    HYPRE_Complex *beta_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex));
    cudaMemcpy(beta_gpu, &beta, sizeof(HYPRE_Complex), cudaMemcpyHostToDevice);
    HYPRE_Int offset = 0; // TODO: Check if this is correct
    HYPRE_Int *offset_gpu = (HYPRE_Int *)gpu_malloc(sizeof(HYPRE_Int));
    cudaMemcpy(offset_gpu, &offset, sizeof(HYPRE_Int), cudaMemcpyHostToDevice);


    HYPRE_MemoryLocation memory_location  = hypre_CSRMatrixMemoryLocation(test_A);
    // printf("Test A Memory location before migrate: %d\n", memory_location);

    // Move data to device
    hypre_CSRMatrixMigrate(test_A, HYPRE_MEMORY_DEVICE);

    memory_location  = hypre_CSRMatrixMemoryLocation(test_A);
    // printf("Test A Memory location after migrate: %d\n", memory_location);

    memory_location  = hypre_VectorMemoryLocation(x);
    // printf("Vec X Memory location after migrate: %d\n", memory_location);

    memory_location  = hypre_VectorMemoryLocation(y);
    // printf("Vec Y Memory location after migrate: %d\n", memory_location);

    HYPRE_Complex *test_A_data = hypre_CSRMatrixData(test_A);

    // For better profiling, optionally allocate memory for sparse matrix to in total lead to dense matrix memory allocation
    // long num_zeros = num_rows * num_cols - nnz;
    // HYPRE_Complex *zeros = (HYPRE_Complex *)malloc(sizeof(HYPRE_Complex) * num_zeros);
    // for(int i = 0; i < num_zeros; i++){
    //     zeros[i] = (HYPRE_Complex)0.0;
    // }
    // HYPRE_Complex *zeros_gpu = (HYPRE_Complex *)gpu_malloc(sizeof(HYPRE_Complex) * num_zeros);
    // cudaMemcpy(zeros_gpu, zeros, sizeof(HYPRE_Complex) * num_zeros, cudaMemcpyHostToDevice);


    printf("Invoking SpMV kernels once before profiling\n");
    // y = alpha * A * x + beta * y
    spmv_amgT_fp64(trans,
                    alpha,
                    test_A,
                    x,
                    beta,
                    y,
                    offset);

    spmv_amgT_fp64_CC(trans,
                    alpha,
                    test_A,
                    x,
                    beta,
                    y,
                    offset);

    spmv_amgT_fp64_TC(trans,
                    alpha,
                    test_A,
                    x,
                    beta,
                    y,
                    offset);


    // -----------Cusparse setup ------------------------------------------------------------
    // HYPRE_Int num_vectors = hypre_VectorNumVectors(x);
    // hypre_CSRMatrix *AT;
    // hypre_CSRMatrix *B;
    // /* SpMV data */
    // size_t bufferSize = 0;
    // char *dBuffer = hypre_CSRMatrixGPUMatSpMVBuffer(test_A);
    // cusparseHandle_t handle = hypre_HandleCusparseHandle(hypre_handle());
    // const cudaDataType data_type = hypre_HYPREComplexToCudaDataType();
    // const cusparseIndexType_t index_type = hypre_HYPREIntToCusparseIndexType();

    // /* Local cusparse descriptor variables */
    // cusparseSpMatDescr_t matA;
    // cusparseDnVecDescr_t vecX, vecY;
    // cusparseDnMatDescr_t matX, matY;

    // B = test_A;

    // /* Create cuSPARSE vector data structures */
    // matA = hypre_CSRMatrixToCusparseSpMat(B, offset);

    // vecX = hypre_VectorToCusparseDnVec(x, 0, num_cols);
    // vecY = hypre_VectorToCusparseDnVec(y, offset, num_rows - offset);

    // if (!dBuffer)
    // {
    //     HYPRE_CUSPARSE_CALL(cusparseSpMV_bufferSize(handle,
    //                                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                                 &alpha,
    //                                                 matA,
    //                                                 vecX,
    //                                                 &beta,
    //                                                 vecY,
    //                                                 data_type,
    //                                                 HYPRE_CUSPARSE_SPMV_ALG,
    //                                                 &bufferSize));

    //     dBuffer = hypre_TAlloc(char, bufferSize, HYPRE_MEMORY_DEVICE);
    //     hypre_CSRMatrixGPUMatSpMVBuffer(test_A) = dBuffer;
    // }

    // cusparseSpMV(handle,
    //                                     CUSPARSE_OPERATION_NON_TRANSPOSE,
    //                                     &alpha,
    //                                     matA,
    //                                     vecX,
    //                                     &beta,
    //                                     vecY,
    //                                     data_type,
    //                                     HYPRE_CUSPARSE_SPMV_ALG,
    //                                     dBuffer);
    // // cudaDeviceSynchronize()
    // hypre_SyncComputeStream(hypre_handle());
    hypre_CSRMatrixMatvecCusparseNewAPI_Cusparse(trans,
                                                alpha,
                                                test_A,
                                                x,
                                                beta,
                                                y,
                                                offset);

    /* Free memory */
    // HYPRE_CUSPARSE_CALL(cusparseDestroySpMat(matA));
    // HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecX));
    // HYPRE_CUSPARSE_CALL(cusparseDestroyDnVec(vecY));
    // ------------ End of Cusparse ------------

    // Copy back and print y
    // cudaMemcpy(y_data, hypre_VectorData(y), sizeof(HYPRE_Complex) * vector_size, cudaMemcpyDeviceToHost);
    // cudaMemcpy(y_data_gpu, y_data, sizeof(HYPRE_Complex) * vector_size, cudaMemcpyHostToDevice);

    // for(int i = 0; i < vector_size; i++){
    //     printf("y[%d]=%lg\n", i, y_data[i]);
    // }

    // Profiling
    double *trial_times = (double*)malloc(num_trials * sizeof(double));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    // Register NVTX string
    nvtxDomainHandle_t profiling_domain = nvtxDomainCreateA("profiling");
    nvtxStringHandle_t profiling_string = nvtxDomainRegisterStringA(profiling_domain, "profiling");
    nvtxEventAttributes_t evtAttr = {0};
    evtAttr.version = NVTX_VERSION;
    evtAttr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    evtAttr.messageType = NVTX_MESSAGE_TYPE_REGISTERED;
    evtAttr.message.registered = profiling_string;


    cudaDeviceSynchronize();
    /* Create thread to gather GPU stats */
    std::thread threadStart( &nvmlClass::getStats,
                            &nvml);  // threadStart starts running
    printf("Profiling...\n");
    sleep(2);
    // Create array to store timing data

    // Track total profiling time
    // time_t start_time = time(NULL);
    // const int MAX_PROFILE_SECONDS = 30; // Cut off after 30 seconds
    // bool timeout = false;

    // nvtxRangePushA("benchmarking");
    nvtxDomainRangePushEx(profiling_domain, &evtAttr);
    for(int i = 0; i < num_trials; i++){
        cudaEventRecord(start);
        if(kernel_type_int == 0){
            spmv_amgT_fp64_TC(trans,
                    alpha,
                    test_A,
                    x,
                    beta,
                    y,
                    offset);
        }else if(kernel_type_int == 1){
            spmv_amgT_fp64_CC(trans,
                    alpha,
                    test_A,
                    x,
                    beta,
                    y,
                    offset);
        }else if(kernel_type_int == 2){
                hypre_CSRMatrixMatvecCusparseNewAPI_Cusparse(trans,
                                                            alpha,
                                                            test_A,
                                                            x,
                                                            beta,
                                                            y,
                                                            offset);
                cudaDeviceSynchronize();
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&milliseconds, start, stop);
        trial_times[i] = milliseconds;
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    // nvtxRangePop();
    // cudaProfilerStop();
    nvtxDomainRangePop(profiling_domain);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    sleep(2);
    /* Create thread to kill GPU stats */
    /* Join both threads to main */
    std::thread threadKill( &nvmlClass::killThread, &nvml );
    threadStart.join( );
    threadKill.join( );
    printf("Profiling done\n");

    // Write timing data to CSV file
    char time_stats_path[1024];
    snprintf(time_stats_path, sizeof(time_stats_path), "%s/%s", dir_path, "timeStats.csv");
    FILE *fp = fopen(time_stats_path, "w");
    fprintf(fp, "trial,time_ms\n");
    for(int i = 0; i < num_trials; i++) {
        fprintf(fp, "%d,%f\n", i, trial_times[i]);
    }
    fclose(fp);
    free(trial_times);


    HYPRE_Finalize();
    nvmlShutdown();
    MPI_Finalize();

    return 0;
}
