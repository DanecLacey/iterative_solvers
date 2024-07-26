#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H
#include "../kernels.hpp"
#include "../sparse_matrix.hpp"
#include "../utility_funcs.hpp"

#ifdef USE_USPMV
#include "../../Ultimate-SpMV/code/interface.hpp"
#endif

template <typename VT>
void gs_iteration_ref_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    VT *tmp,
    VT *D,
    double *b,
    VT *x
){
    double diag_elem = 1.0;
    double sum;

    // std::cout << "sparse_mat->crs_mat->n_rows = " << sparse_mat->crs_mat->n_rows << std::endl;

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
#ifdef DEBUG_MODE_FINE
        std::cout << "x[" << row_idx << "] = " << b[row_idx] << " - " << sum << " / " <<  diag_elem <<std::endl;
#endif
    }
}

template <typename VT>
void gs_iteration_sep_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    double *tmp,
    double *tmp_perm,
    double *D,
    double *b,
    double *x,
    int N
){
    // spmv on strictly upper triangular portion of A to compute tmp <- Ux_{k-1}
#ifdef USE_USPMV
#ifdef USE_AP
    // TODO
#else
    uspmv_omp_scs_cpu<VT, int>(
        sparse_mat->scs_U->C,
        sparse_mat->scs_U->n_chunks,
        &(sparse_mat->scs_U->chunk_ptrs)[0],
        &(sparse_mat->scs_U->chunk_lengths)[0],
        &(sparse_mat->scs_U->col_idxs)[0],
        &(sparse_mat->scs_U->values)[0],
        x,
        tmp_perm
    );
    apply_permutation(tmp, tmp_perm, &(sparse_mat->scs_U->old_to_new_idx)[0], N);
#endif
#else
    spmv_crs_cpu<VT>(tmp, sparse_mat->crs_U, x);
#endif

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
    spltsv_crs<VT>(sparse_mat->crs_L, x, D, tmp);

#ifdef DEBUG_MODE_FINE
    printf("(D+L)^{-1}(b-Ux) = [");
    for(int i = 0; i < N; ++i){
        std::cout << x[i] << ",";
    }
    printf("]\n");
#endif

}

template <typename VT>
void init_gs_structs(
    COOMtxData<VT> *coo_mat,
    SparseMtxFormat<VT> *sparse_mat
){
    COOMtxData<VT> *coo_L = new COOMtxData<VT>;
    COOMtxData<VT> *coo_U = new COOMtxData<VT>;

    split_L_U<VT>(coo_mat, coo_L, coo_U);

#ifdef USE_USPMV
    // Only used for GS kernel
    // TODO: Find a better solution than this crap
    MtxData<VT, int> *mtx_L = new MtxData<VT, int>;
    mtx_L->n_rows = coo_L->n_rows;
    mtx_L->n_cols = coo_L->n_cols;
    mtx_L->nnz = coo_L->nnz;
    mtx_L->is_sorted = true; //TODO
    mtx_L->is_symmetric = false; //TODO
    mtx_L->I = coo_L->I;
    mtx_L->J = coo_L->J;
    mtx_L->values = coo_L->values;
    convert_to_scs<VT, int>(mtx_L, CHUNK_SIZE, SIGMA, sparse_mat->scs_L);

    MtxData<VT, int> *mtx_U = new MtxData<VT, int>;
    mtx_U->n_rows = coo_U->n_rows;
    mtx_U->n_cols = coo_U->n_cols;
    mtx_U->nnz = coo_U->nnz;
    mtx_U->is_sorted = true; //TODO
    mtx_U->is_symmetric = false; //TODO
    mtx_U->I = coo_U->I;
    mtx_U->J = coo_U->J;
    mtx_U->values = coo_U->values;
    convert_to_scs<VT, int>(mtx_U, CHUNK_SIZE, SIGMA, sparse_mat->scs_U);
#endif

    convert_to_crs(coo_L, sparse_mat->crs_L);
    convert_to_crs(coo_U, sparse_mat->crs_U);

    delete coo_L;
    delete coo_U;
}
#endif