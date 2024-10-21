#ifndef KERNELS_H
#define KERNELS_H
#include "sparse_matrix.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <omp.h>

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

template <typename VT>
void mtx_spmv_coo(
    std::vector<double> *y,
    const COOMtxData<VT> *mtx,
    const std::vector<double> *x
    )
{
    int nnz = mtx->nnz;
    if (mtx->I.size() != mtx->J.size() || mtx->I.size() != mtx->values.size()) {
        printf("ERROR: mtx_spmv_coo : sizes of rows, cols, and values differ.\n");
        exit(1);
    }

    // parallel for candidate
    for (int i = 0; i < nnz; ++i) {
        (*y)[mtx->I[i]] += mtx->values[i] * (*x)[mtx->J[i]];
        // std::cout << mtx->values[i] << " * " << (*x)[mtx->J[i]] << " = " << (*y)[mtx->I[i]] << " at idx: " << mtx->I[i] << std::endl; 
    }
}

void vec_spmv_coo(
    std::vector<double> *mult_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
    )
{
    int nrows = vec1->size();

    if(vec1->size() != vec2->size()){
        printf("ERROR: vec_spmv_coo: mismatch in vector sizes.\n");
        exit(1);
    }

    // parallel for candidate
    for (int i = 0; i < nrows; ++i) {
        (*mult_vec)[i] += (*vec1)[i] * (*vec2)[i];
        // std::cout << (*vec)[i] << " * " << (*x)[i]<< " = " << (*y)[i] << " at idx: " << i << std::endl; 
    }
}

template <typename VT1, typename VT2, typename VT3>
void sum_vectors_cpu(
    VT1 *result_vec,
    const VT2 *vec1,
    const VT3 *vec2,
    int N,
    double scale = 1.0
){

    #pragma omp parallel for
    for (int i=0; i<N; i++){
        result_vec[i] = vec1[i] + scale*(vec2[i]);
        // std::cout << static_cast<double>(vec1[i]) << " + ";
        // std::cout << static_cast<double>(scale) << " * ";
        // std::cout << static_cast<double>(vec2[i]) << " = ";
        // std::cout << static_cast<double>(result_vec[i]) << std::endl; 
    }
}

template <typename VT1, typename VT2, typename VT3>
void subtract_vectors_cpu(
    VT1 *result_vec,
    const VT2 *vec1,
    const VT3 *vec2,
    int N,
    double scale = 1.0
){
    // Not an Orphaned directive!
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        result_vec[i] = vec1[i] - scale*vec2[i];
        // std::cout << vec1[i] << " - " << scale << "*" << vec2[i] <<std::endl;
    }
}

template <typename VT, typename FT>
void subtract_residual_cpu(
    VT *residual_vec,
    const VT *vec1,
    const FT *vec2,
    int N,
    double scale = 1.0
){
    // Not an Orphaned directive!
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        residual_vec[i] = vec1[i] - scale*vec2[i];
        // std::cout << vec1[i] << " - " << scale << "*" << vec2[i] <<std::endl;
    }
}

void subtract_vectors_cpu_od(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N,
    double scale
){
    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for
    for (int i = 0; i < N; ++i){
        result_vec[i] = vec1[i] - scale*vec2[i];
        // std::cout << vec1[i] << " - " << scale << "*" << vec2[i] <<std::endl;
    }
}

// Tranpose a dense matrix
void dense_transpose(
    const double *mat,
    double *mat_t,
    int n_rows,
    int n_cols
){
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        for(int col_idx = 0; col_idx < n_cols; ++col_idx){
            mat_t[col_idx*n_rows + row_idx] = mat[row_idx*n_cols + col_idx]; 
        }
    }

}

template <typename VT>
void scale(
    VT *result_vec,
    const VT *vec,
    const double scalar,
    int N
){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        result_vec[i] = vec[i] * scalar;
#ifdef DEBUG_MODE_FINE
        std::cout << "scaling" << std::endl;
        std::cout << result_vec[i] << " = " << vec[i] << " * " << scalar << std::endl;
#endif
    }
}

template <typename VT>
void scale_residual(
    VT *result_vec,
    const VT *res,
    const double scalar,
    int N
){
    #pragma omp parallel for
    for (int i = 0; i < N; ++i){
        result_vec[i] = res[i] * scalar;
#ifdef DEBUG_MODE_FINE
        std::cout << "scaling" << std::endl;
        std::cout << static_cast<double>(result_vec[i]) << " = " << static_cast<double>(res[i]) << " * " << scalar << std::endl;
#endif
    }
}

template <typename VT, typename RT>
void dot(
    const VT *vec1,
    const VT *vec2,
    RT *result,
    int N
){
    RT sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i){
        sum += vec1[i] * vec2[i];
    }
    *result = sum;
}

void dot_od(
    const double *vec1,
    const double *vec2,
    double *partial_sum,
    int N
){
    #pragma omp for
    for (int i = 0; i < N; ++i){
        *partial_sum += vec1[i] * vec2[i];
    }
}

void strided_1_dot(
    const double *vec1,
    const double *vec2,
    double *result,
    int N,
    int stride
){
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i){
        sum += vec1[i*stride] * vec2[i];
    }
    *result = sum;
}

void strided_2_dot(
    const double *vec1,
    const double *vec2,
    double *result,
    int N,
    int stride
){
    double sum = 0;
    // #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i){
        sum += vec1[i] * vec2[i*stride];
        // N = restart_len+1, , stride = restart_len 
    }
    *result = sum;
}

#ifdef __CUDACC__
__global__
void subtract_vectors_gpu(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N
){
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    unsigned int row_idx;

    while(thread_idx + offset < N){
        row_idx = thread_idx + offset;
        result_vec[row_idx] = vec1[row_idx] - vec2[row_idx];

// #ifdef DEBUG_MODE_FINE
        printf("%f = %f - %f at index %i\n", result_vec[row_idx], vec1[row_idx], vec2[row_idx], row_idx); 
// #endif

        offset += stride;
    }
}
#endif

template <typename VT>
void sum_matrices(
    COOMtxData<VT> *sum_mtx,
    const COOMtxData<VT> *mtx1,
    const COOMtxData<VT> *mtx2
){
    // NOTE: Matrices are assumed are to be sorted by row index

    // Traverse through all nz_idx in mtx1 and mtx2, and insert the
    // element with smaller row (and col?) value into the sum_mtx

    // In the case that an element has the same row and column value, 
    // add their values, and inster into the sum_mtx

    // if matrices don't have same dimensions
    if (mtx1->n_rows != mtx2->n_rows || mtx1->n_cols != mtx2->n_cols)
    {
        printf("ERROR: sum_matrices: Matrix dimensions do not match, cannot add.\n");
        exit(1);
    }

    sum_mtx->n_rows = mtx1->n_rows;
    sum_mtx->n_cols = mtx2->n_cols;

    sum_mtx->is_sorted = true;
    sum_mtx->is_symmetric = false; // TODO: How to actually detect this?

    // Just allocate for worst case scenario, in terms of storage
    sum_mtx->I.reserve(mtx1->nnz + mtx2->nnz);
    sum_mtx->J.reserve(mtx1->nnz + mtx2->nnz);
    sum_mtx->values.reserve(mtx1->nnz + mtx2->nnz);

    int nz_idx1 = 0, nz_idx2 = 0; //, nz_idx_sum = 0;
    while (nz_idx1 < mtx1->nnz && nz_idx2 < mtx2->nnz){
        // if mtx2's row and col is smaller (or tiebreaker)
        if (mtx1->I[nz_idx1] > mtx2->I[nz_idx2] ||
            ( mtx1->I[nz_idx1] == mtx2->I[nz_idx2] && mtx1->J[nz_idx1] > mtx2->J[nz_idx2])){

                // insert this smaller value into sum_mtx
                // sum_mtx->I[nz_idx_sum] = mtx2->I[nz_idx2];
                // sum_mtx->J[nz_idx_sum] = mtx2->J[nz_idx2];
                // sum_mtx->values[nz_idx_sum] = mtx2->values[nz_idx2]; 
                sum_mtx->I.push_back(mtx2->I[nz_idx2]);
                sum_mtx->J.push_back(mtx2->J[nz_idx2]);
                sum_mtx->values.push_back(mtx2->values[nz_idx2]);        

                // inc the nz counter for mtx2 and sum_mtx
                ++nz_idx2;        
                // ++nz_idx_sum;
        }
        // else if mtx1's row and col is smaller (or tiebreaker)
        else if (mtx1->I[nz_idx1] < mtx2->I[nz_idx2] ||
        (mtx1->I[nz_idx1] == mtx2->I[nz_idx2] && mtx1->J[nz_idx1] < mtx2->J[nz_idx2])){
            
                // insert this smaller value into sum_mtx
                // sum_mtx->I[nz_idx_sum] = mtx2->I[nz_idx2];
                // sum_mtx->J[nz_idx_sum] = mtx2->J[nz_idx2];
                // sum_mtx->values[nz_idx_sum] = mtx2->values[nz_idx2];  
                sum_mtx->I.push_back(mtx1->I[nz_idx1]);
                sum_mtx->J.push_back(mtx1->J[nz_idx1]);
                sum_mtx->values.push_back(mtx1->values[nz_idx1]);   

                // inc the nz counter for mtx1 and sum_mtx
                ++nz_idx1;     
                // ++nz_idx_sum;
        }
        // else, we actually sum the elements from mtx1 and mtx2!
        else{
            double sum_val = mtx1->values[nz_idx1] + mtx2->values[nz_idx2];
            if(sum_val > 1e-16){ // TODO: Maybe adjust this, just to avoid very small numbers
                // sum_mtx->I[nz_idx_sum] = mtx1->I[nz_idx1]; // Could either take row and col from mtx1 or mtx2
                // sum_mtx->J[nz_idx_sum] = mtx1->J[nz_idx1];
                // sum_mtx->values[nz_idx_sum] = sum_val;     
                sum_mtx->I.push_back(mtx1->I[nz_idx1]); // Could either take row and col from mtx1 or mtx2
                sum_mtx->J.push_back(mtx1->J[nz_idx1]);
                sum_mtx->values.push_back(sum_val);    
            }

            // increment all counters
            ++nz_idx1;
            ++nz_idx2;     
            // ++nz_idx_sum;
        }
    }

        // clean up remaining elements
    while(nz_idx1 < mtx1->nnz){
            sum_mtx->I.push_back(mtx1->I[nz_idx1]);
            sum_mtx->J.push_back(mtx1->J[nz_idx1]);
            sum_mtx->values.push_back(mtx1->values[nz_idx1]);   

        // ++nz_idx_sum;
        ++nz_idx1;
    }

    while(nz_idx2 < mtx2->nnz){
            sum_mtx->I.push_back(mtx2->I[nz_idx2]);
            sum_mtx->J.push_back(mtx2->J[nz_idx2]);
            sum_mtx->values.push_back(mtx2->values[nz_idx2]);   

        // ++nz_idx_sum;
        ++nz_idx2;
    }

    // Sanity checks
    if(nz_idx1 != mtx1->nnz){
        printf("ERROR: sum_matrices: Not all elements from mtx1 accounted for.\n");
        exit(1);
    }
    if(nz_idx2 != mtx2->nnz){
        printf("ERROR: sum_matrices: Not all elements from mtx2 accounted for.\n");
        exit(1);
    }

    sum_mtx->nnz = sum_mtx->values.size();

    // if(sum_mtx->nnz > (nz_idx1 + nz_idx2)){
    //     printf("ERROR: sum_matrices: sum_mtx contains too many nonzeros.\n");
    // }

}

template<typename MT, typename VT>
void spmv_crs_cpu(
    VT *y,
    const CRSMtxData<MT> *crs_mat,
    VT *x
    )
{
    #pragma omp parallel
    {
#ifdef USE_LIKWID
        LIKWID_MARKER_START("native_spmv_benchmark");
#endif
        #pragma omp for schedule(static)
        for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
            double tmp = 0.0;
            #pragma omp simd simdlen(VECTOR_LENGTH) reduction(+:tmp)
            for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
                tmp += crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]];
    // #ifdef DEBUG_MODE_FINE
                // std::cout << static_cast<double>(crs_mat->val[nz_idx]) << " * " << static_cast<double>(x[crs_mat->col[nz_idx]]) << " = " << static_cast<double>(crs_mat->val[nz_idx] * x[crs_mat->col[nz_idx]]) << " at idx: " << row_idx << std::endl; 
    // #endif
            }
            y[row_idx] = tmp;
        }
#ifdef USE_LIKWID
        LIKWID_MARKER_STOP("native_spmv_benchmark");
#endif
    }
}

// TODO: Take from USpMV library later?
#ifdef __CUDACC__
__global__
void spmv_crs_gpu(
    const int d_n_rows,
    const int *d_row_ptr,
    const int *d_col,
    const double *d_val,
    const double *d_x,
    double *d_y
    )
{
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    unsigned int row_idx;

    while(thread_idx + offset < d_n_rows){
        double tmp = 0.0;
        row_idx = thread_idx + offset;

        for(int nz_idx = d_row_ptr[row_idx]; nz_idx < d_row_ptr[row_idx+1]; ++nz_idx){
            tmp += d_val[nz_idx] * d_x[d_col[nz_idx]];
        }

        d_y[row_idx] = tmp;

        // printf("tmp[%i]: %f\n", row_idx, tmp);

        offset += stride;
    }

    // TODO: When is this necessary?
    // __synchthreads();
}
#endif

template <typename VT, typename MT>
void spltsv_crs(
    const CRSMtxData<MT> *crs_L,
    VT *x,
    const MT *D,
    const VT *rhs
)
{
    double sum;
    for(int row_idx = 0; row_idx < crs_L->n_rows; ++row_idx){
        sum = 0.0;
        for(int nz_idx = crs_L->row_ptr[row_idx]; nz_idx < crs_L->row_ptr[row_idx+1]; ++nz_idx){
            sum += crs_L->val[nz_idx] * x[crs_L->col[nz_idx]];
// #ifdef DEBUG_MODE_FINE
//             std::cout << crs_L->val[nz_idx] << " * " << x[crs_L->col[nz_idx]] << " = " << crs_L->val[nz_idx] * x[crs_L->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
// #endif
        }
        x[row_idx] = (rhs[row_idx] - sum)/D[row_idx];
// #ifdef DEBUG_MODE_FINE
//         std::cout << rhs[row_idx] << " - " << sum << " / " << D[row_idx] << " = " << x[row_idx] << " at idx: " << row_idx << std::endl; 
// #endif
    }
}

// C = AB (n x m) = (n x q)(q x m)
void dense_MMM(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
){
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

// C = AB^t (n x q) = (n x m)(m x q)
template <typename VT>
void dense_MMM_t(
    VT *A,
    VT *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
){
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            VT tmp{};
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[k * n_rows_A + i] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

// C = AB??? (n x q) = (n x m)(m x q)
void dense_MMM_t_t(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
){
    // restart-local matrices typically not large enough for parallel for
    // #pragma omp parallel for collapse(2)
    for (int i = 0; i < n_rows_A; ++i) {
        for (int j = 0; j < n_cols_B; ++j) {
            double tmp = 0.0;
            for (int k = 0; k < n_cols_A; ++k) {
                tmp += A[i * n_cols_A + k] * B[k * n_cols_B + j];
            }
            C[i * n_cols_B + j] = tmp;
        }
    }
}

template <typename VT>
VT euclidean_vec_norm_cpu(
    const VT *vec,
    int N
){
    double tmp = 0.0;

    #pragma omp parallel for reduction(+:tmp)
    for(int i = 0; i < N; ++i){
        tmp += vec[i] * vec[i];
        // std::cout << "vec[" << i << "] = " << vec[i] << std::endl;
    }

    return std::sqrt(tmp);
}

template <typename VT>
double infty_vec_norm_cpu(
    const VT *vec,
    int N
){
    double max_abs = 0.;
    double curr_abs;
    for (int i = 0; i < N; ++i){
        // TODO:: Hmmm...
        // curr_abs = std::abs(static_cast<double>(vec[i]));
        curr_abs = (vec[i] >= 0) ? vec[i]  : -1*vec[i];
        if ( curr_abs > max_abs){
            max_abs = curr_abs; 
        }
    }

    return max_abs;
}

#ifdef __CUDACC__
// Begin with all threads unlocked
__device__ int mutex = 0;

__device__ void acquire_semaphore(int *mutex){
    while(atomicCAS(mutex, 0, 1) == 1);
}

__device__ void release_semaphore(int *mutex){
    *mutex = 0;
    __threadfence();
}

__global__
// TODO: known to be buggy/not work
void infty_vec_norm_gpu(
    const double *d_vec,
    double *d_infty_norm,
    double n_rows
){
    const unsigned int thread_idx_in_block = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * blockDim.x;
    const unsigned int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    

    // if(thread_idx + offset < n_rows){

        // Shared memory region, same size as block
        // Will be present on each block
        extern __shared__ double shared_mem[];
        __syncthreads();
        
        for (int i = threadIdx.x; i < THREADS_PER_BLOCK; i += blockDim.x)
            shared_mem[i] = 0.0;
        __syncthreads();

        double tmp = 0.0;
        while(thread_idx + offset < n_rows){
            // NOTE: Absolute values are taken here, and don't need to be done
            // again for comparing block maxes
            tmp = max(abs(tmp), abs(d_vec[thread_idx + offset]));

            offset += stride;
        }

        shared_mem[threadIdx.x] = tmp;

        __syncthreads();

        // Reduction per block
        unsigned int i = blockDim.x/2;
        while(i != 0){
            if(threadIdx.x < i){
                shared_mem[threadIdx.x] = max(abs(shared_mem[threadIdx.x]), abs(shared_mem[threadIdx.x + i]));
            }
            __syncthreads();
            i /= 2;
        } 

        // Begin locks
        __syncthreads();
        if(threadIdx.x == 0){
              acquire_semaphore(&mutex);
        }

        __syncthreads();

        // Critical section
        if(threadIdx.x == 0){
            *d_infty_norm = (*d_infty_norm > shared_mem[0]) ? *d_infty_norm : shared_mem[0];
            printf("%f >? %f\n", *d_infty_norm, shared_mem[0]);
        }

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0){
            release_semaphore(&mutex);
        }

        __syncthreads();
        // End locks

    printf("d_residual_norm = %f\n", *d_infty_norm);
    // }
    
}
#endif

/* Residual here is the distance from A*x_new to b, where the norm
 is the infinity norm: ||A*x_new-b||_infty */
 template <typename MT, typename VT>
void calc_residual_cpu(
    SparseMtxFormat<MT> *sparse_mat,
    VT *x,
    MT *b,
    MT *r,
    VT *tmp,
    VT *tmp_perm,
    int N
){

//  Why do you want to use uspmv for this?
// #ifdef USE_USPMV
//     // uspmv_omp_csr_cpu<double, int>(
//     //     sparse_mat->scs_mat->C,
//     //     sparse_mat->scs_mat->n_chunks,
//     //     &(sparse_mat->scs_mat->chunk_ptrs)[0],
//     //     &(sparse_mat->scs_mat->chunk_lengths)[0],
//     //     &(sparse_mat->scs_mat->col_idxs)[0],
//     //     &(sparse_mat->scs_mat->values)[0],
//     //     x,
//     //     tmp);

//     uspmv_scs_cpu<MT, VT, int>(
//         sparse_mat->scs_mat->C,
//         sparse_mat->scs_mat->n_chunks,
//         &(sparse_mat->scs_mat->chunk_ptrs)[0],
//         &(sparse_mat->scs_mat->chunk_lengths)[0],
//         &(sparse_mat->scs_mat->col_idxs)[0],
//         &(sparse_mat->scs_mat->values)[0],
//         x,
//         tmp_perm
//     );
//     apply_permutation<VT, int>(tmp, tmp_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], N);

// #else
    spmv_crs_cpu<MT, VT>(tmp, sparse_mat->crs_mat, x);
// #endif
    subtract_residual_cpu<MT, VT>(r, b, tmp, N);

#ifdef DEBUG_MODE
    std::cout << "b check" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << static_cast<double>(b[i]) << std::endl;
    }

    std::cout << "tmp check" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << static_cast<double>(tmp[i]) << std::endl;
    }

    std::cout << "Residual check" << std::endl;
    for(int i = 0; i < N; ++i){
        std::cout << static_cast<double>(r[i]) << std::endl;
    }
#endif
}

#ifdef __CUDACC__
void calc_residual_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_x,
    double *d_b,
    double *d_r,
    double *d_tmp,
    int d_n_rows
){
    // TODO: SCS

    // TODO: Block and thread count?
    spmv_crs_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(d_n_rows, d_row_ptr, d_col, d_val, d_x, d_tmp);

    subtract_vectors_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(d_r, d_b, d_tmp, d_n_rows);
}
#endif
#endif /*KERNELS_H*/