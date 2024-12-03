#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cuda_runtime.h"
#include <nvtx3/nvToolsExt.h>
#include "sub_files/mmio_highlevel.h"
#include "sub_files/my_solver.h"
#include "sub_files/subfunction.h"

#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "_hypre_parcsr_mv.h"
#include "seq_mv.h"
#include "ex.h"
#include "_hypre_parcsr_ls.h"

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

int main(int argc, char **argv)
{
    cudaSetDevice(0);
    int m, n, nnzA, isSymmetricA;
    int num_time_iters = 10; //Default
    int *row_ptr; // the csr row pointer array of matrix A
    int *col_idx; // the csr column index array of matrix A
    double *val;  // the csr value array of matrix A
    int *cpu_row_ptr;
    int *cpu_col_idx;
    double *cpu_val;
    double *cpu_bval;

    char *filename_matrix = argv[1];
    char *lastSlash = strrchr(filename_matrix, '/');
    char *lastDot = strrchr(filename_matrix, '.');

    num_time_iters = atoi(argv[2]);
    printf("Num timing iters: %d\n", num_time_iters);

    if (lastSlash != NULL && lastDot != NULL && lastSlash < lastDot)
    {
        // 计算截取的字符串长度
        size_t length = lastDot - (lastSlash + 1);

    }

    char *filename_x = NULL; // the filename of solution vector x

    double *bval;

    int i;
    int myid, num_procs;
    int N;

    int ilower, iupper;
    int local_size, extra;

    int solver_id;
    int vis, print_system, print_system2;

    HYPRE_IJMatrix A;
    HYPRE_ParCSRMatrix parcsr_A;
    HYPRE_IJVector b;
    HYPRE_ParVector par_b;
    HYPRE_IJVector x;
    HYPRE_ParVector par_x;

    HYPRE_Solver solver, precond;

    /* Initialize MPI */
    MPI_Init(&argc, &argv);

    /* Initialize HYPRE */
    HYPRE_Init();
#if defined(HYPRE_USING_GPU)
    /* use vendor implementation for SpxGEMM */
    HYPRE_SetSpGemmUseVendor(1);
#endif
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &num_procs);
    MPI_Comm_rank(comm, &myid);
    int num1;
    int vec1;
    _temp_data pack_data;
    MPI_Datatype newtype;
    MPI_Type_contiguous(6, MPI_INT, &newtype);
    MPI_Type_commit(&newtype);

    /* Default problem parameters */
    // n = 33;
    solver_id = 0;
    vis = 0;
    print_system = 0;
    print_system2 = 0;
#if ReadMMFile
    // elapsed_time(FALSE, 0.);
#if 1
    if (myid == 0)
    {
        mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
        if (m != n)
        {
            printf("Invalid matrix size. %d, %d\n", m, n);
            return 1;
            // phgError(1, "Invalid matrix size.\n");
        }
        printf("Done reading file.\n");

        pack_data.isSymmetricA = isSymmetricA;
        pack_data.m = m;
        pack_data.n = n;
        pack_data.nnzA = nnzA;
        MPI_Bcast(&pack_data, 1, newtype, 0, MPI_COMM_WORLD);

        // x = (double *)malloc(sizeof(double) * n);
        bval = (double *)malloc(sizeof(double) * m);

        // load right-hand side vector b
        // load_b(n, bval, filename_b);
        for (int i = 0; i < n; i++)
            bval[i] = 1.0;
    }
    else
    {
        MPI_Bcast(&pack_data, 1, newtype, 0, MPI_COMM_WORLD);
        isSymmetricA = pack_data.isSymmetricA;
        m = pack_data.m;
        n = pack_data.n;
        nnzA = pack_data.nnzA;
    }
    if (myid != 0)
    {
        row_ptr = (int *)malloc(sizeof(int) * m);
        col_idx = (int *)malloc(sizeof(int) * nnzA);
        val = (double *)malloc(sizeof(double) * nnzA);
        bval = (double *)malloc(sizeof(double) * m);
    }
    // MPI_Barrier(MPI_COMM_WORLD); F
    MPI_Bcast(row_ptr, m + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(col_idx, nnzA, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(val, nnzA, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(bval, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#else
    mmio_allinone(&m, &n, &nnzA, &isSymmetricA, &row_ptr, &col_idx, &val, filename_matrix);
    bval = (double *)malloc(sizeof(double) * m);

    // load right-hand side vector b
    load_b(n, bval, filename_b);
    MPI_Bcast(bval, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
    if (myid == 0)
    {
        printf("Done broadcasting.\n");
    }
    /* Preliminaries: want at least one processor per row */
    N = m; /* global number of rows */
    // printf("myrank = %d, N = %d\n", myid, N);

    /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
    local_size = N / num_procs;
    extra = N - local_size * num_procs;

    // printf("myrank = %d, local_size = %d\n", myid, local_size);

    ilower = local_size * myid;
    ilower += my_min(myid, extra);

    iupper = local_size * (myid + 1);
    iupper += my_min(myid + 1, extra);
    iupper = iupper - 1;

    /* How many rows do I have? */
    local_size = iupper - ilower + 1;

    /* Create the matrix.
      Note that this is a square matrix, so we indicate the row partition
      size twice (since number of rows = number of cols) */
    if (myid == 0)
    {
        printf("Creating HYPRE Matrix...\n");
    }
    HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);

    /* Choose a parallel csr format storage (see the User's Manual) */
    HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);

    /* Initialize before setting coefficients */
    if (myid == 0)
    {
        printf("Initializing HYPRE Matrix...\n");
    }
    HYPRE_IJMatrixInitialize(A);

    /*
    Note that here we are setting one row at a time, though
      one could set all the rows together (see the User's Manual).
   */
    if (myid == 0)
    {
        printf("Allocating data...\n");
    }
    cpu_row_ptr = (int *)gpu_malloc(sizeof(int) * m);
    cpu_col_idx = (int *)gpu_malloc(sizeof(int) * nnzA);
    cpu_val = (double *)gpu_malloc(sizeof(double) * nnzA);
    int *tmp = (int *)gpu_malloc(2 * sizeof(int));
    cudaMemcpy(cpu_row_ptr, row_ptr, m * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cpu_col_idx, col_idx, nnzA * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cpu_val, val, nnzA * sizeof(double), cudaMemcpyHostToDevice);
    if (myid == 0)
    {
        printf("Done memcopying of allocated arrays.\n");
    }
    {
        int _head;


        for (i = ilower; i <= iupper; i++)
        {
            if(i % 100 == 0 && myid == 0)
                printf("Setting matrix values: %d/%d\n", i, iupper);
            /* Set the values for row i */
            _head = row_ptr[i];
            int len = row_ptr[i + 1] - row_ptr[i];
            int row = i;
            tmp[0] = len;
            tmp[1] = i;
            // HYPRE_IJMatrixSetValues(A, 1, &len, &row, &col_idx[_head], &val[_head]);
            HYPRE_IJMatrixSetValues(A, 1, &tmp[0], &tmp[1], &cpu_col_idx[_head], &cpu_val[_head]);
        }
        // free(val);
        // free(row_ptr);
        // free(col_idx);
        // cudaFree(cpu_val);
        // cudaFree(cpu_row_ptr);
        // cudaFree(cpu_col_idx);
    }

    /* Assemble after setting the coefficients */
    printf("begin assemble A\n");
    HYPRE_IJMatrixAssemble(A);
    printf("assemble A finish\n");

    // printf("assemble A finish\n");
    // const char *out_matrix_filename = "mat_out";
    // HYPRE_IJMatrixPrint(A, out_matrix_filename);
    // printf("assemble A finish\n");

    // phgPrintf("  Convert CSR to hypreA time: ");
    // elapsed_time(TRUE, 0.);
#else
    // elapsed_time(FALSE, 0.);
    if (myid == 0)
    {
        mmio_read_crd_size(&m, &n, &nnzA, filename_matrix);
        isSymmetricA = 0;
        if (m != n)
        {
            printf("Invalid matrix size.\n");
            // phgError(1, "Invalid matrix size.\n");
        }

        pack_data.isSymmetricA = isSymmetricA;
        pack_data.m = m;
        pack_data.n = n;
        pack_data.nnzA = nnzA;
        MPI_Bcast(&pack_data, 1, newtype, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Bcast(&pack_data, 1, newtype, 0, MPI_COMM_WORLD);
        isSymmetricA = pack_data.isSymmetricA;
        m = pack_data.m;
        n = pack_data.n;
        nnzA = pack_data.nnzA;
    }

    /* Preliminaries: want at least one processor per row */
    N = m; /* global number of rows */
    /* Each processor knows only of its own rows - the range is denoted by ilower
      and upper.  Here we partition the rows. We account for the fact that
      N may not divide evenly by the number of processors. */
    local_size = N / num_procs;
    extra = N - local_size * num_procs;

    ilower = local_size * myid;
    ilower += my_min(myid, extra);

    iupper = local_size * (myid + 1);
    iupper += my_min(myid + 1, extra);
    iupper = iupper - 1;

    /* How many rows do I have? */
    local_size = iupper - ilower + 1;
    HYPRE_IJMatrixRead("IJ.out.A", MPI_COMM_WORLD, HYPRE_PARCSR, &A);
    // phgPrintf("  Read files to hypreA time: ");
    // elapsed_time(TRUE, 0.);
#endif
    /* Note: for the testing of small problems, one may wish to read
      in a matrix in IJ format (for the format, see the output files
      from the -print_system option).
      In this case, one would use the following routine:
      HYPRE_IJMatrixRead( <filename>, MPI_COMM_WORLD,
                          HYPRE_PARCSR, &A );
      <filename>  = IJ.A.out to read in what has been printed out
      by -print_system (processor numbers are omitted).
      A call to HYPRE_IJMatrixRead is an *alternative* to the
      following sequence of HYPRE_IJMatrix calls:
      Create, SetObjectType, Initialize, SetValues, and Assemble
   */

    /* Get the parcsr matrix object to use */
    HYPRE_IJMatrixGetObject(A, (void **)&parcsr_A);
#if ReadMMFile
    /* Create the rhs and solution */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
    HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(b);

    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    /* Set the rhs values to h^2 and the solution to zero */
    {
        double *rhs_values, *x_values;
        int *rows;

        rhs_values = (double *)gpu_calloc(local_size, sizeof(double));
        x_values = (double *)gpu_calloc(local_size, sizeof(double));
        rows = (int *)gpu_calloc(local_size, sizeof(int));

        for (i = 0; i < local_size; i++)
        {
            rhs_values[i] = bval[ilower + i];
            x_values[i] = 0.0;
            rows[i] = ilower + i;
        }

        printf("Setting b values too...\n");
        HYPRE_IJVectorSetValues(b, local_size, rows, rhs_values);
        HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

        // cudaFree(x_values);
        // cudaFree(rhs_values);
        // cudaFree(rows);
        // cudaFree(bval);
    }


    HYPRE_IJVectorAssemble(b);
#else
    /* Create the rhs and solution */
    HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
    HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
    HYPRE_IJVectorInitialize(x);

    /* Set the rhs values to h^2 and the solution to zero */
    {
        double *x_values;
        int *rows;

        x_values = (double *)calloc(local_size, sizeof(double));
        rows = (int *)calloc(local_size, sizeof(int));

        for (i = 0; i < local_size; i++)
        {
            x_values[i] = 0.0;
            rows[i] = ilower + i;
        }

        HYPRE_IJVectorSetValues(x, local_size, rows, x_values);

        // free(x_values);
        // free(rows);
    }
    HYPRE_IJVectorRead("IJ.out.b", MPI_COMM_WORLD, HYPRE_PARCSR, &b);
#endif
    /*  As with the matrix, for testing purposes, one may wish to read in a rhs:
       HYPRE_IJVectorRead( <filename>, MPI_COMM_WORLD,
                                 HYPRE_PARCSR, &b );
       as an alternative to the
       following sequence of HYPRE_IJVectors calls:
       Create, SetObjectType, Initialize, SetValues, and Assemble
   */
    HYPRE_IJVectorGetObject(b, (void **)&par_b);

    HYPRE_IJVectorAssemble(x);
    HYPRE_IJVectorGetObject(x, (void **)&par_x);

#if 1
    /* PCG with AMG preconditioner */
    {
        int num_iterations;
        int num_level;
        double final_res_norm;
        // 计时
        struct timeval t_start, t_stop;
        struct timeval t1, t2;
        // gettimeofday(&t_start, NULL);

        /* Create solver */
        HYPRE_ParCSRPCGCreate(MPI_COMM_WORLD, &solver);

        /* Set some parameters (See Reference Manual for more parameters) */
        HYPRE_PCGSetMaxIter(solver, 300); /* max iterations */
        HYPRE_PCGSetTol(solver, 1e-5);    /* conv. tolerance */
        HYPRE_PCGSetTwoNorm(solver, 1);   /* use the two norm as the stopping criteria */

        /* Now set up the AMG preconditioner and specify any parameters */
        HYPRE_BoomerAMGCreate(&precond);
        // HYPRE_BoomerAMGSetPrintLevel(precond, 3); /* print amg solution info */
        HYPRE_BoomerAMGSetCoarsenType(precond, 8); // pmis
        HYPRE_BoomerAMGSetMaxLevels(precond, 7);   // 对齐参数 //原10
        HYPRE_BoomerAMGSetMaxRowSum(precond, 0.8); // 对齐参数  //原0.9
        HYPRE_BoomerAMGSetStrongThreshold(precond, 0.25);
        HYPRE_BoomerAMGSetNumFunctions(precond, 3);
        HYPRE_BoomerAMGSetTruncFactor(precond, 0.1); // 对齐参数 //添加
        HYPRE_BoomerAMGSetRelaxType(precond, 18);     // Jacobi
        HYPRE_BoomerAMGSetNumSweeps(precond, 1);
        HYPRE_BoomerAMGSetTol(precond, 1e-20); /* conv. tolerance zero */ // 对齐参数 //原 0.0
        HYPRE_BoomerAMGSetMaxIter(precond, 50);
        HYPRE_BoomerAMGSetMaxCoarseSize(precond, 3);
        HYPRE_BoomerAMGSetCycleNumSweeps(precond, 3, 3);

        HYPRE_Int interp_type = 6;      /* default value */
        HYPRE_Int post_interp_type = 0; /* default value */
        HYPRE_BoomerAMGSetInterpType(precond, interp_type); // extended + i
        HYPRE_BoomerAMGSetRestriction(precond, 0);          // P^T
        HYPRE_BoomerAMGSetPMaxElmts(precond, 4);
        HYPRE_BoomerAMGSetCycleType(precond, 1); // V cycle

        /* Now setup and solve! */
        gettimeofday(&t_start, NULL);
        HYPRE_BoomerAMGSetup(precond, parcsr_A, par_b, par_x);
        gettimeofday(&t_stop, NULL);

        gettimeofday(&t1, NULL);
        HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
        gettimeofday(&t2, NULL);

        double solve_time, total_solve_time;
        total_solve_time = 0.0;

        const char* nvtx_range_name = "AMGSolve_Loop";
        nvtxRangeId_t r1 = nvtxRangeStartA(nvtx_range_name);
        for(int time_iter=0; time_iter<num_time_iters; time_iter++){

            gettimeofday(&t1, NULL);
            HYPRE_BoomerAMGSolve(precond, parcsr_A, par_b, par_x);
            gettimeofday(&t2, NULL);
            solve_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            total_solve_time += solve_time;
            printf("solve_time=%.5lf\n", solve_time);

        }
        nvtxRangeEnd(r1);
        // nvtxRangePop();


        /* Run info - needed logging turned on */
        HYPRE_BoomerAMGGetNumIterations(precond, &num_iterations);
        HYPRE_BoomerAMGGetFinalRelativeResidualNorm(precond, &final_res_norm);

        if (myid == 0)
        {
            printf("\n");
            printf("Iterations = %d\n", num_iterations);
            printf("Final Relative Residual Norm = %e\n", final_res_norm);
            double setup_time = (t_stop.tv_sec - t_start.tv_sec) * 1000.0 + (t_stop.tv_usec - t_start.tv_usec) / 1000.0;
            solve_time = (t2.tv_sec - t1.tv_sec) * 1000.0 + (t2.tv_usec - t1.tv_usec) / 1000.0;
            printf("setup_time=%.5lf\n", setup_time);
            printf("solve_time=%.5lf\n", solve_time);
            printf("time_spmv_sum=%.5lf\n", time_spmv_sum);
            printf("time_spmv=%.5lf\n", time_spmv);
            printf("time_spmv_preprocess=%.5lf\n", time_spmv_preprocess);
            printf("spmv_times=%d\n", spmv_times);
            printf("time_spgemm=%.5lf\n", time_spgemm);
            printf("time_spgemm_preprocess=%.5lf\n", time_spgemm_preprocess);
            // printf("cusparse_spgemm_time=%.5lf\n", cusparse_spgemm_time);
            printf("spgemm_times=%d\n", spgemm_times);
            printf("time_spgemm_all=%.5lf\n", time_spgemm_all);

            printf("csr2bsr_step1=%.5lf\n", csr2bsr_step1);
            printf("csr2bsr_step2=%.5lf\n", csr2bsr_step2);
            printf("csr2bsr_step3=%.5lf\n", csr2bsr_step3);
            printf("bsr2csr_step1=%.5lf\n", bsr2csr_step1);
            printf("bsr2csr_step2=%.5lf\n", bsr2csr_step2);
            printf("bsr2csr_step3=%.5lf\n", bsr2csr_step3);

            printf("Average Solve Time: %f  (N=%d)\n", total_solve_time / num_time_iters, num_time_iters);
            printf("\n");
        }

        /* Destroy solver and preconditioner */
        HYPRE_ParCSRPCGDestroy(solver);
        HYPRE_BoomerAMGDestroy(precond);
    }
#endif

    /* Clean up */
    HYPRE_IJMatrixDestroy(A);
    HYPRE_IJVectorDestroy(b);
    HYPRE_IJVectorDestroy(x);

    /* Finalize HYPRE */
    // HYPRE_Finalize();

    /* Finalize MPI*/
    MPI_Finalize();

    return 0;
}
