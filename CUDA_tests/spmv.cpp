#include "helpers/io_funcs.hpp"
#include "helpers/mmio.h"
#include "helpers/structs.hpp"
#include "helpers/utilities.hpp"
#include "helpers/timing.h"
#include <stdlib.h>
#ifdef __CUDACC__
#include <cusparse.h> 
#endif
#include <string>
#include <omp.h>

#define THREADS_PER_BLOCK 512
#define CPU_BENCH_TIME 1.0
#define GPU_BENCH_TIME 0.005
#define RESTRICT				__restrict__

// #ifdef __CUDACC__

// #define CHECK_CUDA(func)                                                       \
// {                                                                              \
//     cudaError_t status = (func);                                               \
//     if (status != cudaSuccess) {                                               \
//         printf("CUDA API failed at line %d with error: %s (%d)\n",             \
//                __LINE__, cudaGetErrorString(status), status);                  \
//         return EXIT_FAILURE;                                                   \
//     }                                                                          \
// }

// #define CHECK_CUSPARSE(func)                                                   \
// {                                                                              \
//     cusparseStatus_t status = (func);                                          \
//     if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
//         printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
//                __LINE__, cusparseGetErrorString(status), status);              \
//         return EXIT_FAILURE;                                                   \
//     }                                                                          \
// }
// #endif

static void
spmv_scs_cpu(const int C,
             const int n_chunks,
             const int * RESTRICT chunk_ptrs,
             const int * RESTRICT chunk_lengths,
             const int * RESTRICT col_idxs,
             const double * RESTRICT values,
             double * RESTRICT x,
             double * RESTRICT y)
{
    #pragma omp parallel 
    {
        #pragma omp for schedule(static)
        for (int c = 0; c < n_chunks; ++c) {
            double tmp[C];
            for (int i = 0; i < C; ++i) {
                tmp[i] = 0.0;
            }

            int cs = chunk_ptrs[c];

            // TODO: use IT wherever possible
            for (int j = 0; j < chunk_lengths[c]; ++j) {
                for (IT i = 0; i < (IT)C; ++i) {
                    tmp[i] += values[cs + j * (IT)C + i] * x[col_idxs[cs + j * (IT)C + i]];
                }
            }

            for (int i = 0; i < C; ++i) {
                y[c * C + i] = tmp[i];
            }
        }
    }
}

void spmv_crs_cpu(
    const double *val,
    const int *col,
    const int *row_ptr,
    double *y,
    const double *x,
    const int n_rows
    )
{
    double tmp;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        tmp = 0.0;
        for(int nz_idx = row_ptr[row_idx]; nz_idx < row_ptr[row_idx+1]; ++nz_idx){
            tmp += val[nz_idx] * x[col[nz_idx]];
        }
        y[row_idx] = tmp;
    }
}

// TODO: how do
#ifdef __CUDACC__

__global__ static void
spmv_scs_gpu(const long *_C,
         const long *_n_chunks,
         const int * RESTRICT chunk_ptrs,
         const int * RESTRICT chunk_lengths,
         const int * RESTRICT col_idxs,
         const double * RESTRICT values,
         const double * RESTRICT x,
         double * RESTRICT y)
         
{
    long C = *_C;
    long n_chunks = *_n_chunks;
    long row = threadIdx.x + blockDim.x * blockIdx.x;
    int c   = row / C;  // the no. of the chunk
    int idx = row % C;  // index inside the chunk

    if (row < n_chunks * C) {
        double tmp = 0.0;
        int cs = chunk_ptrs[c];

        for (int j = 0; j < chunk_lengths[c]; ++j) {
            tmp += values[cs + j * C + idx] * x[col_idxs[cs + j * C + idx]];
        }

        y[row] = tmp;
    }

}

__global__
void spmv_crs_gpu(
    const double *val,
    const int *col,
    const int *row_ptr,
    double *y,
    const double *x,
    const int n_rows
    )
{
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    double tmp = 0.0;

    // TODO: branch not performant!
    if(thread_idx < n_rows){
        // One thread per row
        for(int nz_idx = row_ptr[thread_idx]; nz_idx < row_ptr[thread_idx+1]; ++nz_idx){
            tmp += val[nz_idx] * x[col[nz_idx]];
        }

        y[thread_idx] = tmp;
    }
}
#endif

int main(int argc, char *argv[]) {
    // Read .mtx file
    if(argc != 5){
        printf("ERROR: Unexpected input. <binary> <kernel> <C> <sigma> <.mtx file>\n");
        exit(1);
    }

    std::string kernel_format = argv[1];
    long C = strtol(argv[2], NULL, 10);
    long sigma = strtol(argv[3], NULL, 10);
    std::string matrix_file_name = argv[4];


    COOMtxData *coo_mat = new COOMtxData;
    CRSMtxData *crs_mat = new CRSMtxData;
    SCSMtxData<double, int> *scs_mat = new SCSMtxData<double, int>;

    printf("Reading matrix...\n");
    read_mtx(coo_mat, matrix_file_name);

    // Write to file //
    // printf("Writing sparsity pattern to output file.\n");
    // std::string file_out_name;
    // file_out_name = "HC_2D_11_L2";

    // for(int idx = 0; idx < coo_mat->nnz; ++idx){
    //     coo_mat->I[idx] += 1;
    //     coo_mat->J[idx] += 1;
    // }

    // mm_write_mtx_crd(
    //     &file_out_name[0], 
    //     coo_mat->n_rows, 
    //     coo_mat->n_cols, 
    //     coo_mat->nnz, 
    //     &(coo_mat->I)[0], 
    //     &(coo_mat->J)[0], 
    //     &(coo_mat->values)[0], 
    //     "MCRG" // TODO: <- make more general, i.e. flexible based on the matrix. Read from original mtx?
    // );
    // exit(0);
    ///////////////////

    int vec_size;
    int nnz;
    double *vals;

    // Used for validation, so always make a copy
    convert_to_crs(coo_mat, crs_mat);

if (kernel_format == "crs"){
    vec_size = crs_mat->n_cols;
    nnz = crs_mat->nnz;
    vals = crs_mat->val;
}
else if (kernel_format == "scs"){
    printf("Converting to SELL-%i-%i...\n", C, sigma);
    convert_to_scs(coo_mat, C, sigma, scs_mat);
    vec_size = scs_mat->n_rows_padded;
    nnz = scs_mat->nnz;
    vals = &(scs_mat->values)[0];
}
else{
    printf("Kernel format not recognized...\n");
    exit(1);
}

    // Initialize x and y
    double *x = new double[vec_size];
    double *y = new double[vec_size];
    generate_x_and_y(x, y, vec_size, nnz, false, vals, 1.1);

#ifdef __CUDACC__
    printf("Moving data to device...\n");
    // Move data to device
    double *d_x; // = new double[vec_size];
    double *d_y; // = new double[vec_size]; // <- dont need to allocate all this space on the host?????
    
    // CRS
    double *d_val;// = new double[crs_mat->nnz];
    int *d_col;// = new int[crs_mat->nnz];
    int *d_row_ptr;// = new int[crs_mat->n_rows + 1];

    // SCS
    long *d_C;
    long *d_n_chunks;
    int *d_chunk_ptrs;
    int *d_chunk_lengths;
    int *d_col_idxs;
    double *d_values;
    int *d_old_to_new_idx;
    int *d_new_to_old_idx;

    if (kernel_format == "crs"){
        cudaMalloc(&d_val, nnz*sizeof(double));
        cudaMalloc(&d_col, nnz*sizeof(int));
        cudaMalloc(&d_row_ptr, (vec_size+1)*sizeof(int));

        cudaMemcpy(d_val, crs_mat->val, nnz*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col, crs_mat->col, nnz*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_ptr, crs_mat->row_ptr, (vec_size+1)*sizeof(int), cudaMemcpyHostToDevice);
    }
    else if (kernel_format == "scs"){
        long n_scs_elements = scs_mat->chunk_ptrs[scs_mat->n_chunks - 1]
                    + scs_mat->chunk_lengths[scs_mat->n_chunks - 1] * scs_mat->C;

        std::cout << "Chunk Occupancy: " << double(scs_mat->nnz) / double(n_scs_elements) << std::endl;
        // exit(0);

        cudaMalloc(&d_C, sizeof(long));
        cudaMalloc(&d_n_chunks, sizeof(long));
        cudaMalloc(&d_chunk_ptrs, (scs_mat->n_chunks + 1)*sizeof(int));
        cudaMalloc(&d_chunk_lengths, scs_mat->n_chunks*sizeof(int));
        cudaMalloc(&d_col_idxs, n_scs_elements*sizeof(int));
        cudaMalloc(&d_values, n_scs_elements*sizeof(double));

        cudaMemcpy(d_C, &scs_mat->C, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_n_chunks, &scs_mat->n_chunks, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_ptrs, &(scs_mat->chunk_ptrs)[0], (scs_mat->n_chunks + 1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_chunk_lengths, &(scs_mat->chunk_lengths)[0], scs_mat->n_chunks*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_col_idxs, &(scs_mat->col_idxs)[0], n_scs_elements*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_values, &(scs_mat->values)[0], n_scs_elements*sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMalloc(&d_x, vec_size*sizeof(double));
    cudaMalloc(&d_y, vec_size*sizeof(double));

    cudaMemcpy(d_x, x, vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, vec_size*sizeof(double), cudaMemcpyHostToDevice);

#ifdef USE_CUSPARSE
    cusparseHandle_t     handle = NULL;
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    float     alpha           = 1.0f;
    float     beta            = 0.0f;

    if (kernel_format == "crs"){
        cusparseCreate(&handle);

        // Create sparse matrix A in CSR format
        cusparseCreateCsr(&matA, crs_mat->n_rows, crs_mat->n_cols, crs_mat->nnz,
                                        d_row_ptr, d_col, d_val,
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
        // Create dense vector X
        cusparseCreateDnVec(&vecX, crs_mat->n_cols, d_x, CUDA_R_64F);
        // Create dense vector y
        cusparseCreateDnVec(&vecY, crs_mat->n_rows, d_y, CUDA_R_64F);
        // allocate an external buffer if needed

        cusparseSpMV_bufferSize(
                                    handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                    CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);
        cudaMalloc(&dBuffer, bufferSize);
    }
#endif

#endif
    printf("Performing SpMVs...\n");
    // Do Spmv (multiple times, to see easier on nsys report)

    float time_elapsed = 0.0;
    float warmup_time_elapsed = 0.0;
    int n_iters = 1000;
    
#ifdef __CUDACC__
    const int num_blocks = (vec_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    cudaEvent_t start, stop, warmup_start, warmup_stop;


/////////////////////////// WARMUP /////////////////////////////
    cudaEventCreate(&warmup_start);
    cudaEventCreate(&warmup_stop);
    

    if (kernel_format == "crs"){
        // do{
            cudaEventRecord(warmup_start);
#ifdef USE_CUSPARSE
            for(int i = 0; i < n_iters; ++i){
                cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
            }
#else
            for(int i = 0; i < n_iters; ++i){

                // Allocate one extra block to catch remaining rows
                spmv_crs_gpu<<< num_blocks, THREADS_PER_BLOCK >>>(
                    d_val,
                    d_col,
                    d_row_ptr,
                    d_y,
                    d_x,
                    vec_size
                );
            }
#endif
            // n_iters *= 2;
            cudaEventRecord(warmup_stop);
            cudaEventSynchronize(warmup_stop);
            cudaEventElapsedTime(&warmup_time_elapsed, warmup_start, warmup_stop);
        // } while (warmup_time_elapsed/1e6 < GPU_BENCH_TIME);
        // n_iters /= 2;
    }
    else if (kernel_format == "scs"){
        // do{
            cudaEventRecord(warmup_start);

            for(int i = 0; i < n_iters; ++i){
                spmv_scs_gpu<<< num_blocks, THREADS_PER_BLOCK >>>(
                    d_C, 
                    d_n_chunks, 
                    d_chunk_ptrs,
                    d_chunk_lengths,
                    d_col_idxs,
                    d_values,
                    d_x,
                    d_y
                );
            }
            // n_iters *= 2;
            cudaEventRecord(warmup_stop);
            cudaEventSynchronize(warmup_stop);
            cudaEventElapsedTime(&warmup_time_elapsed, warmup_start, warmup_stop);
        // } while (warmup_time_elapsed/1e6 < GPU_BENCH_TIME);
        // n_iters /= 2;
    }
////////////////////////////////////////////////////////////////

    // reset from warmup
    n_iters = 1000;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    

    if (kernel_format == "crs"){
        // do{
            cudaEventRecord(start);
#ifdef USE_CUSPARSE
            for(int i = 0; i < n_iters; ++i){
                cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &alpha, matA, vecX, &beta, vecY, CUDA_R_64F,
                                            CUSPARSE_SPMV_ALG_DEFAULT, dBuffer);
            }
#else
            for(int i = 0; i < n_iters; ++i){

                // Allocate one extra block to catch remaining rows
                spmv_crs_gpu<<< num_blocks, THREADS_PER_BLOCK >>>(
                    d_val,
                    d_col,
                    d_row_ptr,
                    d_y,
                    d_x,
                    vec_size
                );
            }
#endif
            // n_iters *= 2;
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed, start, stop);
        // } while (time_elapsed/1e6 < GPU_BENCH_TIME);
        // n_iters /= 2;
    }
    else if (kernel_format == "scs"){
        // do{
            cudaEventRecord(start);
            for(int i = 0; i < n_iters; ++i){
                spmv_scs_gpu<<< num_blocks, THREADS_PER_BLOCK >>>(
                    d_C, 
                    d_n_chunks, 
                    d_chunk_ptrs,
                    d_chunk_lengths,
                    d_col_idxs,
                    d_values,
                    d_x,
                    d_y
                );
            }
            // n_iters *= 2;
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&time_elapsed, start, stop);
        // } while (time_elapsed/1e6 < GPU_BENCH_TIME);
        // n_iters /= 2;
    }

    // Convert milliseconds to seconds
    time_elapsed /= 1e3;

#ifdef USE_CUSPARSE
    // destroy matrix/vector descriptors
    cusparseDestroySpMat(matA);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecY);
    cusparseDestroy(handle);
#endif
#else
/////////////////////////// WARMUP /////////////////////////////
    double warmup_start, warmup_end;

    if (kernel_format == "crs"){
        // do{
            warmup_start = getTimeStamp();
            for(int k = 0; k < n_iters; ++k) {
                spmv_crs_cpu(
                    crs_mat->val,
                    crs_mat->col,
                    crs_mat->row_ptr,
                    y,
                    x,
                    vec_size
                );
            }
            // n_iters *= 2;
            warmup_time_elapsed = getTimeStamp() - warmup_start;
        // } while (warmup_time_elapsed < CPU_BENCH_TIME);
        // n_iters /= 2;
    }
    else if (kernel_format == "scs"){
        // do{
            warmup_start = getTimeStamp();
            for(int k = 0; k < n_iters; ++k){
                spmv_scs_cpu(
                    scs_mat->C, 
                    scs_mat->n_chunks, 
                    &(scs_mat->chunk_ptrs)[0],
                    &(scs_mat->chunk_lengths)[0],
                    &(scs_mat->col_idxs)[0],
                    &(scs_mat->values)[0],
                    x,
                    y
                );
            }
            // n_iters *= 2;
            warmup_time_elapsed = getTimeStamp() - warmup_start;
        // } while (warmup_time_elapsed < CPU_BENCH_TIME);
        // n_iters /= 2;
    }
////////////////////////////////////////////////////////////////

    n_iters = 1000; // reset from warmup
    double start, end;
    
    if (kernel_format == "crs"){
        // do{
            start = getTimeStamp();
            for(int k = 0; k < n_iters; ++k) {
                spmv_crs_cpu(
                    crs_mat->val,
                    crs_mat->col,
                    crs_mat->row_ptr,
                    y,
                    x,
                    vec_size
                );
            }
            // n_iters *= 2;
            time_elapsed = getTimeStamp() - start;
        // } while (time_elapsed < CPU_BENCH_TIME);
        // n_iters /= 2;
    }
    else if (kernel_format == "scs"){
        // do{
            start = getTimeStamp();
            for(int k = 0; k < n_iters; ++k){
                spmv_scs_cpu(
                    scs_mat->C, 
                    scs_mat->n_chunks, 
                    &(scs_mat->chunk_ptrs)[0],
                    &(scs_mat->chunk_lengths)[0],
                    &(scs_mat->col_idxs)[0],
                    &(scs_mat->values)[0],
                    x,
                    y
                );
            }
            // n_iters *= 2;
            time_elapsed = getTimeStamp() - start;
        // } while (time_elapsed < CPU_BENCH_TIME);
        // n_iters /= 2;
    }
#endif

double duration_total_s = time_elapsed;
double duration_kernel_s = duration_total_s / n_iters;
double perf_gflops = nnz * 2.0
                    / duration_kernel_s
                    / 1e9;                   // Only count usefull flops

#ifdef __CUDACC__
    printf("Copying data back to host...\n");
    // Move result data back to host
    cudaMemcpy(y, d_y, vec_size*sizeof(double), cudaMemcpyDeviceToHost);
#endif

    printf("Validating results...\n");
    if (kernel_format == "scs"){
        double *unpermmed_y = new double [scs_mat->n_rows];
        apply_permutation(unpermmed_y, y, &(scs_mat->old_to_new_idx)[0], scs_mat->n_rows);
        std::swap(unpermmed_y, y);
        delete unpermmed_y;
    }
    // Collect error by comparing against MKL
    validate_dp_result(crs_mat, x, y);

    // Report results
    printf("Time elapsed for computation: %1.16f\n", time_elapsed);
    printf("Performance for %i iterations: %f GF/s\n", n_iters, perf_gflops);

#ifdef __CUDACC__
    cudaFree(d_x);
    cudaFree(d_y);

    if (kernel_format == "crs"){
        cudaFree(d_val);
        cudaFree(d_col);
        cudaFree(d_row_ptr);
    }
    else if (kernel_format == "scs"){
        cudaFree(d_chunk_ptrs);
        cudaFree(d_chunk_lengths);
        cudaFree(d_col_idxs);
        cudaFree(d_val);
        cudaFree(d_old_to_new_idx);
        cudaFree(d_new_to_old_idx);
    }

#endif

    delete coo_mat;
    delete crs_mat;
    delete scs_mat;
    delete x;
    delete y;
}