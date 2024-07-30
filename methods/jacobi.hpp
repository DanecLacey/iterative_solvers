#ifndef JACOBI_H
#define JACOBI_H
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"
#include "../utility_funcs.hpp"

#ifdef USE_USPMV
#include "../../Ultimate-SpMV/code/interface.hpp"
#endif

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
    int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    unsigned int row_idx;

    while(thread_idx + offset < n_rows){
        row_idx = thread_idx + offset;
        double adjusted_x = d_x_new[row_idx] - d_D[row_idx] * d_x_old[row_idx];
        d_x_new[row_idx] = (d_b[row_idx] - adjusted_x) / d_D[row_idx];

        offset += stride;
    }
}
#endif


template <typename VT>
void jacobi_normalize_x_cpu(
    VT *x_new,
    const VT *x_old,
    const VT *D,
    const VT *rhs,
    int n_rows
){
    VT adjusted_x;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        adjusted_x = x_new[row_idx] - D[row_idx] * x_old[row_idx];
        x_new[row_idx] = (rhs[row_idx] - adjusted_x)/ D[row_idx];
#ifdef DEBUG_MODE_FINE
            std::cout << rhs[row_idx] << " - " << adjusted_x << " / " << D[row_idx] << " = " << x_new[row_idx] << " at idx: " << row_idx << std::endl; 
#endif
    }
}


template <typename VT>
void jacobi_iteration_ref_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    VT *D,
    VT *b,
    VT *x_old,
    VT *x_new // treat like y
){
    VT diag_elem{};

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        VT sum{};
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == sparse_mat->crs_mat->col[nz_idx]){
                diag_elem = sparse_mat->crs_mat->val[nz_idx];
                // if (std::abs(diag_elem) < 1e16)
                //     diag_elem = 1.0; // What to do in this case?
            }
            else{
                sum += sparse_mat->crs_mat->val[nz_idx] * x_old[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
#ifdef DEBUG_MODE_FINE
        std::cout << "x_new[" << row_idx << "] = " << b[row_idx] << " - " << sum << " / " <<  diag_elem <<std::endl;
#endif
    }
}


/*
    I would think this would allow the easiest library integration, since the SpMV kernel is the same.
    Except here, you would need some way to avoid opening and closing the two parallel regions.
*/
template <typename VT>
void jacobi_iteration_sep_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    VT *D,
    VT *b,
    VT *x_old,
    VT *x_old_perm,
    VT *x_new,
    VT *x_new_perm,
    int n_rows
){
#ifdef USE_USPMV
#ifdef USE_AP
    uspmv_omp_scs_ap_cpu<VT, int>(
        sparse_mat->scs_mat_hp->n_chunks,
        sparse_mat->scs_mat_hp->C,
        &(sparse_mat->scs_mat_hp->chunk_ptrs)[0],
        &(sparse_mat->scs_mat_hp->chunk_lengths)[0],
        &(sparse_mat->scs_mat_hp->col_idxs)[0],
        &(sparse_mat->scs_mat_hp->values)[0],
        x_old,
        x_new_perm,
        sparse_mat->scs_mat_lp->n_chunks,
        sparse_mat->scs_mat_lp->C,
        &(sparse_mat->scs_mat_lp->chunk_ptrs)[0],
        &(sparse_mat->scs_mat_lp->chunk_lengths)[0],
        &(sparse_mat->scs_mat_lp->col_idxs)[0],
        &(sparse_mat->scs_mat_lp->values)[0]
    );
    apply_permutation(x_new, x_new_perm, &(sparse_mat->scs_mat_hp->old_to_new_idx)[0], n_rows);

#else
    uspmv_omp_scs_cpu<VT, int>(
        sparse_mat->scs_mat->C,
        sparse_mat->scs_mat->n_chunks,
        &(sparse_mat->scs_mat->chunk_ptrs)[0],
        &(sparse_mat->scs_mat->chunk_lengths)[0],
        &(sparse_mat->scs_mat->col_idxs)[0],
        &(sparse_mat->scs_mat->values)[0],
        x_old,
        x_new_perm
    );
    apply_permutation(x_new, x_new_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_rows);

#endif
#else
    spmv_crs_cpu<VT>(x_new, sparse_mat->crs_mat, x_old);
#endif
    // account for diagonal element in sum, RHS, and division 
    jacobi_normalize_x_cpu<VT>(x_new, x_old, D, b, n_rows);
}

#ifdef __CUDACC__
__global__
void jacobi_iteration_ref_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new,
    int n_rows
){
    double diag_elem = 0;
    double sum = 0;

    const unsigned int thread_idx_in_block = threadIdx.x;
    const unsigned int block_offset = blockIdx.x * blockDim.x;
    const unsigned int thread_idx = block_offset + thread_idx_in_block;
    const unsigned int stride = gridDim.x * blockDim.x; // <- equiv. to total num threads
    unsigned int offset = 0;
    unsigned int row_idx;

    while (thread_idx + offset < n_rows){
        row_idx = thread_idx + offset;
        sum = 0;
        for(int nz_idx = d_row_ptr[row_idx]; nz_idx < d_row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == d_col[nz_idx]){
                diag_elem = d_val[nz_idx];
            }
            else{
                sum += d_val[nz_idx] * d_x_old[d_col[nz_idx]];
#ifdef DEBUG_MODE_FINE
                printf("%f * %f = %f at index %i\n", d_val[nz_idx], d_x_old[d_col[nz_idx]], d_val[nz_idx] * d_x_old[d_col[nz_idx]], row_idx); 
#endif
            }
        }

        d_x_new[row_idx] = (d_b[row_idx] - sum) / diag_elem;

        offset += stride;        
    }
}


/*
    I would think this would allow the easiest library integration, since the SpMV kernel is the same.
    Except here, you would need some way to avoid opening and closing the two parallel regions.
*/
void jacobi_iteration_sep_gpu(
    int d_n_rows,
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new
){
    int n_rows = d_n_rows;
#ifdef USE_USPMV
    if(CHUNK_SIZE > 1 || SIGMA > 1){
        printf("ERROR: SCS not yet supported in jacobi_iteration_sep_gpu.\n");
    }
    uspmv_csr_gpu<double, int><<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(
        d_n_rows,
        d_row_ptr,
        d_col,
        d_val,
        d_x_old,
        d_x_new);
#else
    spmv_crs_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_n_rows, d_row_ptr, d_col, d_val, d_x_old, d_x_new);
#endif
    // account for diagonal element in sum, RHS, and division 
    jacobi_normalize_x_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_x_new, d_x_old, d_D, d_b, d_n_rows);
}
#else

#endif

#endif