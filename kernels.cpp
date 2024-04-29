#include "kernels.hpp"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <omp.h>

void mtx_spmv_coo(
    std::vector<double> *y,
    const COOMtxData *mtx,
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

void sum_vectors(
    std::vector<double> *result_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
){
#ifdef DEBUG_MODE
    if(vec1->size() != vec2->size()){
        printf("ERROR: sum_vectors: mismatch in vector sizes.\n");
        exit(1);
    }
    if(vec1->size() == 0){
        printf("ERROR: sum_vectors: zero size vectors.\n");
        exit(1);
    }
#endif

    #pragma omp parallel for
    for (int i=0; i<vec1->size(); i++){
        (*result_vec)[i] = (*vec1)[i] + (*vec2)[i];
    }
}

void subtract_vectors_cpu(
    std::vector<double> *result_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
){
#ifdef DEBUG_MODE
    if(vec1->size() != vec2->size()){
        printf("ERROR: sum_vectors: mismatch in vector sizes.\n");
        exit(1);
    }
    if(vec1->size() == 0){
        printf("ERROR: sum_vectors: zero size vectors.\n");
        exit(1);
    }
#endif
    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for
    for (int i=0; i<vec1->size(); i++){
        (*result_vec)[i] = (*vec1)[i] - (*vec2)[i];
    }
}

#ifdef __CUDACC__
__global__
void subtract_vectors_gpu(
    double *result_vec,
    const double *vec1,
    const double *vec2
){
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int i = block_offset + thread_idx_in_block;

    if(i < N){(*result_vec)[i] = (*vec1)[i] - (*vec2)[i];}
}
#endif

void sum_matrices(
    COOMtxData *sum_mtx,
    const COOMtxData *mtx1,
    const COOMtxData *mtx2
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

void spmv_crs_cpu(
    std::vector<double> *y,
    const CRSMtxData *crs_mat,
    const std::vector<double> *x
    )
{
    double tmp;

    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule (static)
    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
        tmp = 0.0;
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            tmp += crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]];
#ifdef DEBUG_MODE
            std::cout << crs_mat->val[nz_idx] << " * " << (*x)[crs_mat->col[nz_idx]] << " = " << crs_mat->val[nz_idx] * (*x)[crs_mat->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
#endif
        }
        (*y)[row_idx] = tmp;
    }
}

// TODO: Take from USpMV library later?
#ifdef __CUDACC__
__global__
void spmv_crs_gpu(
    const double *d_val,
    const int *d_col,
    const int *d_row_ptr,
    double *d_y,
    const double *d_x,
    const int d_n_rows
    )
{
    // Idea is for each thread to be responsible for one row
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int i = block_offset + thread_idx_in_block;

    if(i < n_rows){
        double tmp = 0.0;

        for(int nz_idx = d_row_ptr[i]; nz_idx < d_row_ptr[i+1]; ++nz_idx){
            tmp += d_val[nz_idx] * d_x[d_col[nz_idx]];
        }

        d_y[i] = tmp;
    }

    // TODO: When is this necessary?
    // __synchthreads();
}
#endif

void jacobi_normalize_x_cpu(
    std::vector<double> *x_new,
    const std::vector<double> *x_old,
    const std::vector<double> *D,
    const std::vector<double> *b,
    int n_rows
){
    double adjusted_x;

    // Orphaned directive: Assumed already called within a parallel region
    #pragma omp for schedule (static)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        adjusted_x = (*x_new)[row_idx] - (*D)[row_idx] * (*x_old)[row_idx];
        (*x_new)[row_idx] = ((*b)[row_idx] - adjusted_x)/ (*D)[row_idx];
#ifdef DEBUG_MODE
            std::cout << (*b)[row_idx] << " - " << adjusted_x << " / " << (*D)[row_idx] << " = " << (*x_new)[row_idx] << " at idx: " << row_idx << std::endl; 
#endif
    }
}

#ifdef __CUDACC__
__global__
void jacobi_normalize_x_gpu(
    double *d_x_new,
    const double *d_x_old,
    const double *d_D,
    const double *d_b,
    int n_rows
){
    int thread_idx_in_block = threadIdx.x;
    int block_offset = blockIdx.x*blockDim.x;
    int i = block_offset + thread_idx_in_block;

    if(i < n_rows){
        double adjusted_x = d_x_new[i] - d_D[i] * d_x_old[i];
        d_x_new[i] = (d_b[i] - adjusted_x) / d_D[i];
    }
}
#endif

void spltsv_crs(
    const CRSMtxData *crs_L,
    std::vector<double> *x,
    const std::vector<double> *D,
    const std::vector<double> *b_Ux
)
{
    double sum;

    for(int row_idx = 0; row_idx < crs_L->n_rows; ++row_idx){
        sum = 0.0;
        for(int nz_idx = crs_L->row_ptr[row_idx]; nz_idx < crs_L->row_ptr[row_idx+1]; ++nz_idx){
            sum += crs_L->val[nz_idx] * (*x)[crs_L->col[nz_idx]];
#ifdef DEBUG_MODE
            std::cout << crs_L->val[nz_idx] << " * " << (*x)[crs_L->col[nz_idx]] << " = " << crs_L->val[nz_idx] * (*x)[crs_L->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
#endif
        }
        (*x)[row_idx] = ((*b_Ux)[row_idx] - sum)/(*D)[row_idx];
    }
}

double infty_vec_norm_cpu(
    const std::vector<double> *vec
){
    double max_abs = 0.;
    double curr_abs;
    for (int i = 0; i < vec->size(); ++i){
        curr_abs = std::abs((*vec)[i]);
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
    

    // Is this necessary?
    // if(thread_idx < N){

        // Shared memory region, same size as block
        // Will be present on each block
        __shared__ double shared_mem[THREADS_PER_BLOCK];

        double tmp = 0.0;
        while(thread_idx + offset < N){
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
                shared_mem[threadIdx.x] = max(shared_mem[threadIdx.x], shared_mem[threadIdx.x + i]);
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
        }

        __threadfence();
        __syncthreads();

        if (threadIdx.x == 0){
            release_semaphore(&mutex);
        }

        __syncthreads();
        // End locks
    // }

}
#endif

/* Residual here is the distance from A*x_new to b, where the norm
 is the infinity norm: ||A*x_new-b||_infty */
void calc_residual_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *x,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *tmp
){
    //Unpack args 
    #pragma omp parallel
    {
#ifdef USE_USPMV
        spmv_omp_scs<double, int>(
            sparse_mat->scs_mat->C, // C
            sparse_mat->scs_mat->n_chunks,
            &(sparse_mat->scs_mat->chunk_ptrs)[0],
            &(sparse_mat->scs_mat->chunk_lengths)[0],
            &(sparse_mat->scs_mat->col_idxs)[0],
            &(sparse_mat->scs_mat->values)[0],
            &(*x)[0],
            &(*tmp)[0]);
#else
        spmv_crs_cpu(tmp, sparse_mat->crs_mat, x);
#endif
        subtract_vectors_cpu(r, b, tmp);
    }
}

#ifdef __CUDACC__
__global__
void calc_residual_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_x,
    double *d_r,
    double *d_b,
    double *d_tmp,
    int d_n_rows
){
    // TODO: SCS

    // TODO: Block and thread count?
    spmv_crs_gpu<<<1,1>>>(d_val, d_col, d_row_ptr, d_tmp, d_x, d_n_rows);

    subtract_vectors_gpu<<<1,1>>>(d_r, d_b, d_tmp);
}
#endif