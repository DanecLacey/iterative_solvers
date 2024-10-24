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
    VT *b,
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
    VT *tmp, 
    VT *tmp_perm, 
    VT *D, 
    VT *b, 
    VT *x,
#ifdef USE_USPMV
#ifdef USE_AP
    double *tmp_dp,
    double *tmp_perm_dp,
    // double *D_dp,
    // double *b_dp,
    double *x_dp,
    float *tmp_sp,
    float *tmp_perm_sp,
    // float *D_sp,
    // float *b_sp,
    float *x_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *tmp_hp,
    _Float16 *tmp_perm_hp,
    // _Float16 *D_hp,
    // _Float16 *b_hp,
    _Float16 *x_hp,
#endif
#endif
#endif
    int n_rows
){
    // spmv on strictly upper triangular portion of A to compute tmp <- Ux_{k-1}
#ifdef USE_USPMV
    execute_uspmv<VT, int>(
        &(sparse_mat->scs_U->C),
        &(sparse_mat->scs_U->n_chunks),
        sparse_mat->scs_U->chunk_ptrs.data(),
        sparse_mat->scs_U->chunk_lengths.data(),
        sparse_mat->scs_U->col_idxs.data(),
        sparse_mat->scs_U->values.data(),
        x, //input
        tmp_perm, //output
#ifdef USE_AP
        &(sparse_mat->scs_U_dp->C),
        &(sparse_mat->scs_U_dp->n_chunks),
        sparse_mat->scs_U_dp->chunk_ptrs.data(),
        sparse_mat->scs_U_dp->chunk_lengths.data(),
        sparse_mat->scs_U_dp->col_idxs.data(),
        sparse_mat->scs_U_dp->values.data(),
        x_dp,
        tmp_perm_dp,
        &(sparse_mat->scs_U_sp->C),
        &(sparse_mat->scs_U_sp->n_chunks),
        sparse_mat->scs_U_sp->chunk_ptrs.data(),
        sparse_mat->scs_U_sp->chunk_lengths.data(),
        sparse_mat->scs_U_sp->col_idxs.data(),
        sparse_mat->scs_U_sp->values.data(),
        x_sp,
        tmp_perm_sp,
#ifdef HAVE_HALF_MATH
        &(sparse_mat->scs_U_hp->C),
        &(sparse_mat->scs_U_hp->n_chunks),
        sparse_mat->scs_U_hp->chunk_ptrs.data(),
        sparse_mat->scs_U_hp->chunk_lengths.data(),
        sparse_mat->scs_U_hp->col_idxs.data(),
        sparse_mat->scs_U_hp->values.data(),
        x_hp,
        tmp_perm_hp,
#endif
#endif
        AP_VALUE_TYPE
    );

#ifdef DEBUG_MODE_FINE
#ifdef USE_AP
    printf("Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp_perm_dp[i]) << ",";
    }
    printf("]\n");
#else
    printf("Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp_perm[i]) << ",";
    }
    printf("]\n");
#endif
#endif

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        apply_permutation(tmp_dp, tmp_perm_dp, &(sparse_mat->scs_U_dp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        apply_permutation(tmp_sp, tmp_perm_sp, &(sparse_mat->scs_U_sp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        apply_permutation(tmp_hp, tmp_perm_hp, &(sparse_mat->scs_U_hp->old_to_new_idx)[0], n_rows);
#endif
    }
#else
    apply_permutation(tmp, tmp_perm, &(sparse_mat->scs_U->old_to_new_idx)[0], n_rows);
#endif
#else
    spmv_crs_cpu<VT>(tmp, sparse_mat->crs_U, x);
#endif

#ifdef DEBUG_MODE_FINE
#ifdef USE_AP
    printf("Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp_dp[i]) << ",";
    }
    printf("]\n");
#else
    printf("Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp[i]) << ",";
    }
    printf("]\n");
#endif
#endif

// Perform b - Ux
#ifdef USE_USPMV
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        subtract_vectors_cpu<double, VT>(tmp_dp, b, tmp_dp, n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        subtract_vectors_cpu<float, VT>(tmp_sp, b, tmp_sp, n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        subtract_vectors_cpu<_Float16, VT>(tmp_hp, b, tmp_hp, n_rows);
#endif
    }
#endif
#endif
    subtract_vectors_cpu<VT, VT, VT, VT>(tmp, b, tmp, n_rows);

#ifdef DEBUG_MODE_FINE
#ifdef USE_AP
    printf("b-Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp_dp[i]) << ",";
    }
    printf("]\n");
#else
    printf("b-Ux = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(tmp[i]) << ",";
    }
    printf("]\n");
#endif
#endif

// sparse_mat->crs_L->print();

    // performs the lower triangular solve (L+D)x_k=b-Ux_{k-1}
    // NOTE: We keep in CRS, as there isn't a benefit to ltsv in SELL-C-simga
// #ifdef USE_USPMV
// #ifdef USE_AP
//     if(xstr(WORKING_PRECISION) == "double")
//         spltsv_crs<double>(sparse_mat->crs_L_dp, x_dp, D_dp, tmp_dp);
//     else if(xstr(WORKING_PRECISION) == "float")
//         spltsv_crs<float>(sparse_mat->crs_L_sp, x_sp, D_sp, tmp_sp);
//     else if(xstr(WORKING_PRECISION) == "half"){
// #ifdef HAVE_HALF_MATH
//         spltsv_crs<_Float16>(sparse_mat->crs_L_hp, x_hp, D_hp, tmp_hp);
// #endif
//     }
// #endif
// #endif

#ifdef USE_USPMV
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        spltsv_crs<double, VT>(sparse_mat->crs_L, x_dp, D, tmp_dp);
    else if(xstr(WORKING_PRECISION) == "float")
        spltsv_crs<float, VT>(sparse_mat->crs_L, x_sp, D, tmp_sp);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        spltsv_crs<_Float16, VT>(sparse_mat->crs_L, x_hp, D, tmp_hp);
#endif
    }
#endif
#endif

    spltsv_crs<VT, VT>(sparse_mat->crs_L, x, D, tmp);

#ifdef DEBUG_MODE_FINE
#ifdef USE_AP
    printf("(D+L)^{-1}(b-Ux) = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(x_dp[i]) << ",";
    }
    printf("]\n");
#else
    printf("(D+L)^{-1}(b-Ux) = [");
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(x[i]) << ",";
    }
    printf("]\n");
#endif
#endif

    // exit(1);

}

template <typename VT>
void allocate_gs_structs(
#ifdef USE_AP
    MtxData<double, int> *mtx_mat_dp,
    MtxData<float, int> *mtx_mat_sp,
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *mtx_mat_hp,
#endif
    char* ap_value_type,
#endif
    COOMtxData<double> *coo_mat,
    SparseMtxFormat<VT> *sparse_mat
){
    COOMtxData<double> *coo_L = new COOMtxData<double>;
    COOMtxData<double> *coo_U = new COOMtxData<double>;
#ifdef USE_USPMV
#ifdef USE_AP
    MtxData<double, int> *mtx_L_dp = new MtxData<double, int>;
    MtxData<double, int> *mtx_U_dp = new MtxData<double, int>;
    MtxData<float, int> *mtx_L_sp = new MtxData<float, int>;
    MtxData<float, int> *mtx_U_sp = new MtxData<float, int>;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *mtx_L_hp = new MtxData<_Float16, int>;
    MtxData<_Float16, int> *mtx_U_hp = new MtxData<_Float16, int>;
#endif
#endif
#endif

    split_L_U<double, double>(
        &coo_mat->is_sorted,
        &coo_mat->is_symmetric,
        &coo_mat->n_cols,
        &coo_mat->n_rows,
        &coo_mat->nnz,
        &coo_mat->I,
        &coo_mat->J,
        &coo_mat->values,
        &coo_L->is_sorted,
        &coo_L->is_symmetric,
        &coo_L->n_cols,
        &coo_L->n_rows,
        &coo_L->nnz,
        &coo_L->I,
        &coo_L->J,
        &coo_L->values,
        &coo_U->is_sorted,
        &coo_U->is_symmetric,
        &coo_U->n_cols,
        &coo_U->n_rows,
        &coo_U->nnz,
        &coo_U->I,
        &coo_U->J,
        &coo_U->values
    );

#ifdef USE_AP
    split_L_U<double, double>(
        &mtx_mat_dp->is_sorted,
        &mtx_mat_dp->is_symmetric,
        &mtx_mat_dp->n_cols,
        &mtx_mat_dp->n_rows,
        &mtx_mat_dp->nnz,
        &mtx_mat_dp->I,
        &mtx_mat_dp->J,
        &mtx_mat_dp->values,
        &mtx_L_dp->is_sorted,
        &mtx_L_dp->is_symmetric,
        &mtx_L_dp->n_cols,
        &mtx_L_dp->n_rows,
        &mtx_L_dp->nnz,
        &mtx_L_dp->I,
        &mtx_L_dp->J,
        &mtx_L_dp->values,
        &mtx_U_dp->is_sorted,
        &mtx_U_dp->is_symmetric,
        &mtx_U_dp->n_cols,
        &mtx_U_dp->n_rows,
        &mtx_U_dp->nnz,
        &mtx_U_dp->I,
        &mtx_U_dp->J,
        &mtx_U_dp->values
    );

    split_L_U<float, float>(
        &mtx_mat_sp->is_sorted,
        &mtx_mat_sp->is_symmetric,
        &mtx_mat_sp->n_cols,
        &mtx_mat_sp->n_rows,
        &mtx_mat_sp->nnz,
        &mtx_mat_sp->I,
        &mtx_mat_sp->J,
        &mtx_mat_sp->values,
        &mtx_L_sp->is_sorted,
        &mtx_L_sp->is_symmetric,
        &mtx_L_sp->n_cols,
        &mtx_L_sp->n_rows,
        &mtx_L_sp->nnz,
        &mtx_L_sp->I,
        &mtx_L_sp->J,
        &mtx_L_sp->values,
        &mtx_U_sp->is_sorted,
        &mtx_U_sp->is_symmetric,
        &mtx_U_sp->n_cols,
        &mtx_U_sp->n_rows,
        &mtx_U_sp->nnz,
        &mtx_U_sp->I,
        &mtx_U_sp->J,
        &mtx_U_sp->values
    );

#ifdef HAVE_HALF_MATH
    split_L_U<_Float16, _Float16>(
        &mtx_mat_hp->is_sorted,
        &mtx_mat_hp->is_symmetric,
        &mtx_mat_hp->n_cols,
        &mtx_mat_hp->n_rows,
        &mtx_mat_hp->nnz,
        &mtx_mat_hp->I,
        &mtx_mat_hp->J,
        &mtx_mat_hp->values,
        &mtx_L_hp->is_sorted,
        &mtx_L_hp->is_symmetric,
        &mtx_L_hp->n_cols,
        &mtx_L_hp->n_rows,
        &mtx_L_hp->nnz,
        &mtx_L_hp->I,
        &mtx_L_hp->J,
        &mtx_L_hp->values,
        &mtx_U_hp->is_sorted,
        &mtx_U_hp->is_symmetric,
        &mtx_U_hp->n_cols,
        &mtx_U_hp->n_rows,
        &mtx_U_hp->nnz,
        &mtx_U_hp->I,
        &mtx_U_hp->J,
        &mtx_U_hp->values
    );
#endif
#endif

#ifdef USE_USPMV
    // Only used for GS kernel
    // TODO: Find a better solution than this crap
    MtxData<double, int> *mtx_L = new MtxData<double, int>;
    mtx_L->n_rows = coo_L->n_rows;
    mtx_L->n_cols = coo_L->n_cols;
    mtx_L->nnz = coo_L->nnz;
    mtx_L->is_sorted = true; //TODO
    mtx_L->is_symmetric = false; //TODO
    mtx_L->I = coo_L->I;
    mtx_L->J = coo_L->J;
    mtx_L->values = coo_L->values;

    convert_to_scs<double, VT, int>(mtx_L, CHUNK_SIZE, SIGMA, sparse_mat->scs_L);

// Don't need L, only U for spmv
// #ifdef USE_AP
//     convert_to_scs<double, double, int>(mtx_L_dp, CHUNK_SIZE, SIGMA, sparse_mat->scs_L_dp);
//     convert_to_scs<float, float, int>(mtx_L_sp, CHUNK_SIZE, SIGMA, sparse_mat->scs_L_sp);
// #ifdef HAVE_HALF_MATH
//     convert_to_scs<_Float16, _Float16, int>(mtx_L_hp, CHUNK_SIZE, SIGMA, sparse_mat->scs_L_hp);
// #endif
// #endif

    MtxData<double, int> *mtx_U = new MtxData<double, int>;
    mtx_U->n_rows = coo_U->n_rows;
    mtx_U->n_cols = coo_U->n_cols;
    mtx_U->nnz = coo_U->nnz;
    mtx_U->is_sorted = true; //TODO
    mtx_U->is_symmetric = false; //TODO
    mtx_U->I = coo_U->I;
    mtx_U->J = coo_U->J;
    mtx_U->values = coo_U->values;
    convert_to_scs<double, VT, int>(mtx_U, CHUNK_SIZE, SIGMA, sparse_mat->scs_U);
#ifdef USE_AP
    convert_to_scs<double, double, int>(mtx_U_dp, CHUNK_SIZE, SIGMA, sparse_mat->scs_U_dp);
    convert_to_scs<float, float, int>(mtx_U_sp, CHUNK_SIZE, SIGMA, sparse_mat->scs_U_sp);
#ifdef HAVE_HALF_MATH
    convert_to_scs<_Float16, _Float16, int>(mtx_U_hp, CHUNK_SIZE, SIGMA, sparse_mat->scs_U_hp);
#endif
#endif
#endif

    // Just convenient to have a CRS copy too
    convert_to_crs<double, VT>(
        &coo_L->n_rows,
        &coo_L->n_cols,
        &coo_L->nnz,
        &(coo_L->I),
        &(coo_L->J),
        &(coo_L->values), 
        sparse_mat->crs_L
    );
    convert_to_crs<double, VT>(
        &coo_U->n_rows,
        &coo_U->n_cols,
        &coo_U->nnz,
        &(coo_U->I),
        &(coo_U->J),
        &(coo_U->values), 
        sparse_mat->crs_U
    );

// Don't need AP L matrix
// #ifdef USE_AP
//     convert_to_crs<double, double>(
//         &mtx_L_dp->n_rows,
//         &mtx_L_dp->n_cols,
//         &mtx_L_dp->nnz,
//         &(mtx_L_dp->I),
//         &(mtx_L_dp->J),
//         &(mtx_L_dp->values), 
//         sparse_mat->crs_L_dp
//     );
//     convert_to_crs<double, double>(
//         &mtx_U_dp->n_rows,
//         &mtx_U_dp->n_cols,
//         &mtx_U_dp->nnz,
//         &(mtx_U_dp->I),
//         &(mtx_U_dp->J),
//         &(mtx_U_dp->values), 
//         sparse_mat->crs_U_dp
//     );
//     convert_to_crs<float, float>(
//         &mtx_L_sp->n_rows,
//         &mtx_L_sp->n_cols,
//         &mtx_L_sp->nnz,
//         &(mtx_L_sp->I),
//         &(mtx_L_sp->J),
//         &(mtx_L_sp->values), 
//         sparse_mat->crs_L_sp
//     );
//     convert_to_crs<float, float>(
//         &mtx_U_sp->n_rows,
//         &mtx_U_sp->n_cols,
//         &mtx_U_sp->nnz,
//         &(mtx_U_sp->I),
//         &(mtx_U_sp->J),
//         &(mtx_U_sp->values), 
//         sparse_mat->crs_U_sp
//     );
//     convert_to_crs<_Float16, _Float16>(
//         &mtx_L_hp->n_rows,
//         &mtx_L_hp->n_cols,
//         &mtx_L_hp->nnz,
//         &(mtx_L_hp->I),
//         &(mtx_L_hp->J),
//         &(mtx_L_hp->values), 
//         sparse_mat->crs_L_hp
//     );
//     convert_to_crs<_Float16, _Float16>(
//         &mtx_U_hp->n_rows,
//         &mtx_U_hp->n_cols,
//         &mtx_U_hp->nnz,
//         &(mtx_U_hp->I),
//         &(mtx_U_hp->J),
//         &(mtx_U_hp->values), 
//         sparse_mat->crs_U_hp
//     );
// #endif

    // ALL WRONG. You have to take your other copies after the matrix splitting
//     // Copy crs_L and crs_U to all precisions
// #ifdef USE_USPMV
// #ifdef USE_AP
//     sparse_mat->crs_L_dp->n_rows = sparse_mat->crs_L->n_rows;
//     sparse_mat->crs_L_dp->n_cols = sparse_mat->crs_L->n_cols;
//     sparse_mat->crs_L_dp->nnz = sparse_mat->crs_L->nnz;
//     sparse_mat->crs_L_sp->n_rows = sparse_mat->crs_L->n_rows;
//     sparse_mat->crs_L_sp->n_cols = sparse_mat->crs_L->n_cols;
//     sparse_mat->crs_L_sp->nnz = sparse_mat->crs_L->nnz;
// #ifdef USE_AP
//     sparse_mat->crs_L_hp->n_rows = sparse_mat->crs_L->n_rows;
//     sparse_mat->crs_L_hp->n_cols = sparse_mat->crs_L->n_cols;
//     sparse_mat->crs_L_hp->nnz = sparse_mat->crs_L->nnz;
// #endif
//     // Allocate room for arrays in copy structs
//     sparse_mat->crs_L_dp->row_ptr = new int[sparse_mat->crs_L->n_rows + 1];
//     sparse_mat->crs_L_sp->row_ptr = new int[sparse_mat->crs_L->n_rows + 1];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_L_hp->row_ptr = new int[sparse_mat->crs_L->n_rows + 1];
// #endif
//     // Copy data
//     #pragma omp parallel for
//     for(int i = 0; i < sparse_mat->crs_L->n_rows + 1; ++i){
//         sparse_mat->crs_L_dp->row_ptr[i] = sparse_mat->crs_L->row_ptr[i];
//         sparse_mat->crs_L_sp->row_ptr[i] = sparse_mat->crs_L->row_ptr[i];
// #ifdef HAVE_HALF_MATH
//         sparse_mat->crs_L_hp->row_ptr[i] = sparse_mat->crs_L->row_ptr[i];
// #endif
//     }
//     // Allocate room for arrays in copy structs
//     sparse_mat->crs_L_dp->col = new int[sparse_mat->crs_L->nnz];
//     sparse_mat->crs_L_sp->col = new int[sparse_mat->crs_L->nnz];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_L_hp->col = new int[sparse_mat->crs_L->nnz];
// #endif
//     sparse_mat->crs_L_dp->val = new double[sparse_mat->crs_L->nnz];
//     sparse_mat->crs_L_sp->val = new float[sparse_mat->crs_L->nnz];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_L_hp->val = new _Float16[sparse_mat->crs_L->nnz];
// #endif
//     // Copy data
//     #pragma omp parallel for
//     for(int i = 0; i < sparse_mat->crs_L->nnz; ++i){
//         sparse_mat->crs_L_dp->col[i] = sparse_mat->crs_L->col[i];
//         sparse_mat->crs_L_dp->val[i] = static_cast<double>(sparse_mat->crs_L->val[i]);
//         sparse_mat->crs_L_sp->col[i] = sparse_mat->crs_L->col[i];
//         sparse_mat->crs_L_sp->val[i] = static_cast<float>(sparse_mat->crs_L->val[i]);
// #ifdef HAVE_HALF_MATH
//         sparse_mat->crs_L_hp->col[i] = sparse_mat->crs_L->col[i];
//         sparse_mat->crs_L_hp->val[i] = static_cast<_Float16>(sparse_mat->crs_L->val[i]);
// #endif
//     }

//     sparse_mat->crs_U_dp->n_rows = sparse_mat->crs_U->n_rows;
//     sparse_mat->crs_U_dp->n_cols = sparse_mat->crs_U->n_cols;
//     sparse_mat->crs_U_dp->nnz = sparse_mat->crs_U->nnz;
//     sparse_mat->crs_U_sp->n_rows = sparse_mat->crs_U->n_rows;
//     sparse_mat->crs_U_sp->n_cols = sparse_mat->crs_U->n_cols;
//     sparse_mat->crs_U_sp->nnz = sparse_mat->crs_U->nnz;
// #ifdef USE_AP
//     sparse_mat->crs_U_hp->n_rows = sparse_mat->crs_U->n_rows;
//     sparse_mat->crs_U_hp->n_cols = sparse_mat->crs_U->n_cols;
//     sparse_mat->crs_U_hp->nnz = sparse_mat->crs_U->nnz;
// #endif
//     // Allocate room for arrays in copy structs
//     sparse_mat->crs_U_dp->row_ptr = new int[sparse_mat->crs_U->n_rows + 1];
//     sparse_mat->crs_U_sp->row_ptr = new int[sparse_mat->crs_U->n_rows + 1];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_U_hp->row_ptr = new int[sparse_mat->crs_U->n_rows + 1];
// #endif
//     // Copy data
//     #pragma omp parallel for
//     for(int i = 0; i < sparse_mat->crs_U->n_rows + 1; ++i){
//         sparse_mat->crs_U_dp->row_ptr[i] = sparse_mat->crs_U->row_ptr[i];
//         sparse_mat->crs_U_sp->row_ptr[i] = sparse_mat->crs_U->row_ptr[i];
// #ifdef HAVE_HALF_MATH
//         sparse_mat->crs_U_hp->row_ptr[i] = sparse_mat->crs_U->row_ptr[i];
// #endif
//     }
//     // Allocate room for arrays in copy structs
//     sparse_mat->crs_U_dp->col = new int[sparse_mat->crs_U->nnz];
//     sparse_mat->crs_U_sp->col = new int[sparse_mat->crs_U->nnz];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_U_hp->col = new int[sparse_mat->crs_U->nnz];
// #endif
//     sparse_mat->crs_U_dp->val = new double[sparse_mat->crs_U->nnz];
//     sparse_mat->crs_U_sp->val = new float[sparse_mat->crs_U->nnz];
// #ifdef HAVE_HALF_MATH
//     sparse_mat->crs_U_hp->val = new _Float16[sparse_mat->crs_U->nnz];
// #endif
//     // Copy data
//     #pragma omp parallel for
//     for(int i = 0; i < sparse_mat->crs_U->nnz; ++i){
//         sparse_mat->crs_U_dp->col[i] = sparse_mat->crs_U->col[i];
//         sparse_mat->crs_U_dp->val[i] = static_cast<double>(sparse_mat->crs_U->val[i]);
//         sparse_mat->crs_U_sp->col[i] = sparse_mat->crs_U->col[i];
//         sparse_mat->crs_U_sp->val[i] = static_cast<float>(sparse_mat->crs_U->val[i]);
// #ifdef HAVE_HALF_MATH
//         sparse_mat->crs_U_hp->col[i] = sparse_mat->crs_U->col[i];
//         sparse_mat->crs_U_hp->val[i] = static_cast<_Float16>(sparse_mat->crs_U->val[i]);
// #endif
//     }
// #endif
// #endif

    delete coo_L;
    delete coo_U;
#ifdef USE_USPMV
#ifdef USE_AP
    delete mtx_L_dp;
    delete mtx_U_dp;
    delete mtx_L_sp;
    delete mtx_U_sp;
#ifdef HAVE_HALF_MATH
    delete mtx_L_hp;
    delete mtx_U_hp;
#endif
#endif
#endif
}
#endif