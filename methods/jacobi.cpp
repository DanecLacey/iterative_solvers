#include "../kernels.hpp"
#include "../structs.hpp"

#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif

void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new // treat like y
){
    double diag_elem = 0.0;
    double sum = 0;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        sum = 0;
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
void jacobi_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new,
    int N
){
    int n_rows = N; // <- make more flexible

    #pragma omp parallel
    {
#ifdef USE_USPMV
        // uspmv_omp_scs_cpu<double, int>(
        uspmv_omp_scs_cpu(
            sparse_mat->scs_mat->C,
            sparse_mat->scs_mat->n_chunks,
            &(sparse_mat->scs_mat->chunk_ptrs)[0],
            &(sparse_mat->scs_mat->chunk_lengths)[0],
            &(sparse_mat->scs_mat->col_idxs)[0],
            &(sparse_mat->scs_mat->values)[0],
            x_old,
            x_new);

        n_rows = sparse_mat->scs_mat->n_rows;
        // TODO: not sure which is correct
        // n_rows = sparse_mat->scs_mat->n_rows_padded;
#else
        spmv_crs_cpu(x_new, sparse_mat->crs_mat, x_old);
#endif
        // account for diagonal element in sum, RHS, and division 
        jacobi_normalize_x_cpu(x_new, x_old, D, b, N);
    }
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
#endif