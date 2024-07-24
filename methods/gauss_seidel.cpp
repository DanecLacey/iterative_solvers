#include "../kernels.hpp"
#include "../structs.hpp"

#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x
){
    double diag_elem = 1.0;
    double sum;

    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        sum = 0.0;
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == sparse_mat->crs_mat->col[nz_idx]){
                diag_elem = sparse_mat->crs_mat->val[nz_idx];
            }
            else{
                sum += sparse_mat->crs_mat->val[nz_idx] * x[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        x[row_idx] = (b[row_idx] - sum) / diag_elem;
    }
}


void gs_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
){
    #pragma omp parallel
    {
        // spmv on strictly upper triangular portion of A to compute tmp <- Ux_{k-1}
// #ifdef USE_USPMV
//          TODO: Bug with upper triangular SpMV in USpMV library
//         spmv_omp_scs<double, int>(
//             sparse_mat->scs_U->C,
//             sparse_mat->scs_U->n_chunks,
//             &(sparse_mat->scs_U->chunk_ptrs)[0],
//             &(sparse_mat->scs_U->chunk_lengths)[0],
//             &(sparse_mat->scs_U->col_idxs)[0],
//             &(sparse_mat->scs_U->values)[0],
//             &(*x)[0],
//             &(*x)[0]);
// #else
        // trspmv_crs(tmp, crs_U, x); // <- TODO: Could you benefit from a triangular spmv?
        spmv_crs_cpu(tmp, sparse_mat->crs_U, x);
// #endif

#ifdef DEBUG_MODE_FINE
        printf("Ux = [");
        for(int i = 0; i < N; ++i){
            std::cout << tmp[i] << ",";
        }
        printf("]\n");
#endif

        // subtract b to compute tmp <- b-Ux_{k-1}
        subtract_vectors_cpu(tmp, b, tmp, N);
#ifdef DEBUG_MODE_FINE
        printf("b-Ux = [");
        for(int i = 0; i < N; ++i){
            std::cout << tmp[i] << ",";
        }
        printf("]\n");
#endif

        // performs the lower triangular solve (L+D)x_k=b-Ux_{k-1}
        #pragma omp master
        {
            spltsv_crs(sparse_mat->crs_L, x, D, tmp);
        }
#ifdef DEBUG_MODE_FINE
        printf("(D+L)^{-1}(b-Ux) = [");
        for(int i = 0; i < N; ++i){
            std::cout << x[i] << ",";
        }
        printf("]\n");
#endif
    
    }
}