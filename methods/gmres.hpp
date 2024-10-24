#ifndef GMRES_H
#define GMRES_H

#include <iomanip>
#include <cmath>

#include "../stopwatch.hpp"
#include "../kernels.hpp"
#include "../utility_funcs.hpp"
#include "../sparse_matrix.hpp"

#ifdef USE_USPMV
#include "../../Ultimate-SpMV/code/interface.hpp"
#endif

template <typename VT>
struct gmresArgs
{
    double beta;
    VT *init_v;
    VT *V;
#ifdef USE_USPMV
#ifdef USE_AP
    double *V_dp;
    float *V_sp;
#ifdef HAVE_HALF_MATH
    _Float16 *V_hp;
#endif
#endif
#endif
    VT *Vy;
    VT *H;
    VT *H_tmp;
    VT *J;
    VT *R;
    VT *Q;
    VT *Q_copy;
    VT *g;
    VT *g_copy;
    int restart_count;
    int restart_length;
};

template <typename VT>
void apply_gmres_preconditioner(
    std::string preconditioner_type,
    SparseMtxFormat<VT> *sparse_mat,
    VT *vec,
    VT *rhs,
    VT *D,
    int n_cols){
    if(!preconditioner_type.empty()){
        if(preconditioner_type == "jacobi"){
            // jacobi_iteration_sep_cpu(
            //     sparse
            // )
            // TODO
        }
        else if(preconditioner_type == "gauss-seidel"){
            spltsv_crs<VT>(sparse_mat->crs_L, vec, D, rhs);
        }
        else{
            printf("ERROR: apply_gmres_preconditioner: preconditioner_type not recognized.\n");
            exit(1);
        }
    }
}

template <typename VT>
void gmres_iteration_ref_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    Timers *timers,
    std::string preconditioner_type,
    double beta, // <- why does this not need to be a pointer dereference?
    VT *D,
    VT *V,
    VT *H,
    VT *H_tmp,
    VT *J,
    VT *Q,
    VT *Q_copy,
    VT *w,
    VT *w_perm,
    VT *R,
    VT *g,
    VT *g_copy,
    VT *b,
    VT *x,
#ifdef USE_USPMV
#ifdef USE_AP
    double *V_dp,
    double *w_dp,
    double *w_perm_dp,
    float *V_sp,
    float *w_sp,
    float *w_perm_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *V_hp,
    _Float16 *w_hp,
    _Float16 *w_perm_hp,
#endif
#endif
#endif
    int n_rows,
    int restart_count,
    int iter_count,
    VT *residual_norm,
    int restart_len
){
    // NOTES:
    // - The orthonormal vectors in V are stored as row vectors

    iter_count -= restart_count*restart_len;

#ifdef DEBUG_MODE
    // Lazy way to account for restarts
    std::cout << "gmres solve iter_count = " << iter_count << std::endl;
    std::cout << "gmres solve restart_count = " << restart_count << std::endl;
    // Tolerance for validation checks
    double tol=1e-14;
    int fixed_width = 12;
#endif

    ///////////////////// SpMV Step /////////////////////
    // w_k = A*v_k (SpMV)
    // NOTE: We don't time the permutations in the case of USpMV, 
    // as that is an aspect of the implementation and not necessary for Sell-C-Sigma
    // timers->gmres_spmv_wtime->start_stopwatch();
// #ifdef USE_USPMV
//     execute_spmv<VT, int>(
//         sparse_mat->scs_mat_dp->C,
//         sparse_mat->scs_mat_dp->n_chunks,
//         sparse_mat->scs_mat_dp->chunk_ptrs,
//         sparse_mat->scs_mat_dp->chunk_lengths,
//         sparse_mat->scs_mat_dp->col_idxs,
//         sparse_mat->scs_mat_dp->values,
//         sparse_mat->scs_mat_sp->C,
//         sparse_mat->scs_mat_sp->n_chunks,
//         sparse_mat->scs_mat_sp->chunk_ptrs,
//         sparse_mat->scs_mat_sp->chunk_lengths,
//         sparse_mat->scs_mat_sp->col_idxs,
//         sparse_mat->scs_mat_sp->values,
// #ifdef HAVE_HALF_MATH
//         sparse_mat->scs_mat_hp->C,
//         sparse_mat->scs_mat_hp->n_chunks,
//         sparse_mat->scs_mat_hp->chunk_ptrs,
//         sparse_mat->scs_mat_hp->chunk_lengths,
//         sparse_mat->scs_mat_hp->col_idxs,
//         sparse_mat->scs_mat_hp->values,
// #endif
//         &V[iter_count*n_rows],
// #ifdef USE_AP
//         w,
//         AP_VALUE_TYPE
// #else
//         w
// #endif
//     );
// #else
//     spmv_crs_cpu<VT>(&V[iter_count*n_rows], sparse_mat->crs_mat, w);
// #endif
//     timers->gmres_spmv_wtime->end_stopwatch();

//     // If using USPMV, then need to permute x
// #ifdef USE_USPMV
// #ifdef USE_AP
//     apply_permutation(w, w_perm, &(sparse_mat->scs_mat_dp->old_to_new_idx)[0], n_rows);
// #else
//     apply_permutation(w, w_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_rows);
// #endif
// #endif

    timers->gmres_spmv_wtime->start_stopwatch();

#ifdef USE_USPMV
    execute_uspmv<VT, int>(
        &(sparse_mat->scs_mat->C),
        &(sparse_mat->scs_mat->n_chunks),
        sparse_mat->scs_mat->chunk_ptrs.data(),
        sparse_mat->scs_mat->chunk_lengths.data(),
        sparse_mat->scs_mat->col_idxs.data(),
        sparse_mat->scs_mat->values.data(),
        &V[iter_count*n_rows], //input
        w_perm, //output
#ifdef USE_AP
        &(sparse_mat->scs_mat_dp->C),
        &(sparse_mat->scs_mat_dp->n_chunks),
        sparse_mat->scs_mat_dp->chunk_ptrs.data(),
        sparse_mat->scs_mat_dp->chunk_lengths.data(),
        sparse_mat->scs_mat_dp->col_idxs.data(),
        sparse_mat->scs_mat_dp->values.data(),
        &V_dp[iter_count*n_rows], //input
        w_perm_dp, //output
        &(sparse_mat->scs_mat_sp->C),
        &(sparse_mat->scs_mat_sp->n_chunks),
        sparse_mat->scs_mat_sp->chunk_ptrs.data(),
        sparse_mat->scs_mat_sp->chunk_lengths.data(),
        sparse_mat->scs_mat_sp->col_idxs.data(),
        sparse_mat->scs_mat_sp->values.data(),
        &V_sp[iter_count*n_rows], //input
        w_perm_sp, //output
#ifdef HAVE_HALF_MATH
        &(sparse_mat->scs_mat_hp->C),
        &(sparse_mat->scs_mat_hp->n_chunks),
        sparse_mat->scs_mat_hp->chunk_ptrs.data(),
        sparse_mat->scs_mat_hp->chunk_lengths.data(),
        sparse_mat->scs_mat_hp->col_idxs.data(),
        sparse_mat->scs_mat_hp->values.data(),
        &V_hp[iter_count*n_rows], //input
        w_perm_hp, //output
#endif
#endif
        AP_VALUE_TYPE
    );

    timers->gmres_spmv_wtime->end_stopwatch();

    // Permute rows back
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        apply_permutation(w_dp, w_perm_dp, &(sparse_mat->scs_mat_dp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        apply_permutation(w_sp, w_perm_sp, &(sparse_mat->scs_mat_sp->old_to_new_idx)[0], n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        apply_permutation(w_hp, w_perm_hp, &(sparse_mat->scs_mat_hp->old_to_new_idx)[0], n_rows);
#endif
    }
#else
    apply_permutation(w, w_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_rows);
#endif
#else
    // If you're not using USPMV, then just perform this spmv
    spmv_crs_cpu<VT>(w, sparse_mat->crs_mat, &V[iter_count*n_rows]);
    timers->gmres_spmv_wtime->end_stopwatch();
#endif

    timers->gmres_apply_preconditioner_wtime->start_stopwatch();
    // TODO: Be careful with the precisions here!
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        apply_gmres_preconditioner<double>(preconditioner_type, sparse_mat, w_dp, w_dp, D, n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        apply_gmres_preconditioner<float>(preconditioner_type, sparse_mat, w_sp, w_sp, D, n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        apply_gmres_preconditioner<_Float16>(preconditioner_type, sparse_mat, w_hp, w_hp, D, n_rows);
#endif
    }
#else
    apply_gmres_preconditioner<VT>(preconditioner_type, sparse_mat, w, w, D, n_rows);
#endif
    timers->gmres_apply_preconditioner_wtime->end_stopwatch();


#ifdef DEBUG_MODE
    std::cout << "w = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << static_cast<double>(w[i]) << ", ";
        }
    std::cout << "]" << std::endl;
#endif

    ///////////////////// Orthogonalization Step /////////////////////
    timers->gmres_orthog_wtime->start_stopwatch();

    // For all v \in V
    timers->gmres_mgs_wtime->start_stopwatch();
    for(int j = 0; j <= iter_count; ++j){
        // h_ij <- (w,v)
#ifdef FINE_TIMERS
        timers->gmres_mgs_dot_wtime->start_stopwatch();
#endif

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        dot(w_dp, &V_dp[j*n_rows], &H[iter_count + j*restart_len] , n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        dot(w_sp, &V_sp[j*n_rows], &H[iter_count + j*restart_len] , n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        dot(w_hp, &V_hp[j*n_rows], &H[iter_count + j*restart_len] , n_rows);
#endif
    }
#else
        dot(w, &V[j*n_rows], &H[iter_count + j*restart_len] , n_rows);
#endif

#ifdef FINE_TIMERS
        timers->gmres_mgs_dot_wtime->end_stopwatch(); 
#endif

#ifdef DEBUG_MODE
        std::cout << "h_" << j << "_" << iter_count << " = ";
        std::cout << static_cast<double>(H[iter_count + j*restart_len]) << std::endl;
#endif

        // w_j <- w_j - h_ij*v_k
#ifdef FINE_TIMERS
        timers->gmres_mgs_sub_wtime->start_stopwatch();
#endif

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        subtract_vectors_cpu(w_dp, w_dp, &V_dp[j*n_rows], n_rows, H[iter_count + j*restart_len]); 
    else if(xstr(WORKING_PRECISION) == "float")
        subtract_vectors_cpu(w_sp, w_sp, &V_sp[j*n_rows], n_rows, H[iter_count + j*restart_len]); 
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        subtract_vectors_cpu(w_hp, w_hp, &V_hp[j*n_rows], n_rows, H[iter_count + j*restart_len]); 
#endif
    }
#else
        subtract_vectors_cpu(w, w, &V[j*n_rows], n_rows, H[iter_count + j*restart_len]); 
#endif

        
#ifdef FINE_TIMERS
        timers->gmres_mgs_sub_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
        std::cout << "adjusted_w_" << j << "_rev  = [";
            for(int i = 0; i < n_rows; ++i){
                std::cout << static_cast<double>(w[i]) << ", ";
            }
        std::cout << "]" << std::endl;
#endif
    }
    timers->gmres_mgs_wtime->end_stopwatch();


#ifdef DEBUG_MODE
    std::cout << "V" << " = [\n";
        for(int row_idx = 0; row_idx < restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < n_rows; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(V[(n_rows*row_idx) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Save norm to Hessenberg matrix subdiagonal H[k+1,k]
    // H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w, n_rows);

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w_dp, n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w_sp, n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w_hp, n_rows);
#endif
    }
#else
        H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w, n_rows);
#endif


#ifdef DEBUG_MODE
    std::cout << "H" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(H[(restart_len*row_idx) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Normalize the new orthogonal vector v <- v/H[k+1,k]
    // scale(&V[(iter_count+1)*n_rows], w, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        scale(&V_dp[(iter_count+1)*n_rows], w_dp, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);
    else if(xstr(WORKING_PRECISION) == "float")
        scale(&V_sp[(iter_count+1)*n_rows], w_sp, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        scale(&V_hp[(iter_count+1)*n_rows], w_hp, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);
#endif
    }
#else
        scale(&V[(iter_count+1)*n_rows], w, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);
#endif

#ifdef DEBUG_MODE
    std::cout << "v_" << iter_count+1 << " = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << std::setw(fixed_width);
            std::cout << static_cast<double>(V[(iter_count+1)*n_rows + i]) << ", ";
        }
    std::cout << "]" << std::endl;
#endif

#ifndef USE_AP
#ifdef DEBUG_MODE
    // Sanity check: Check if all basis vectors in V are orthonormal
    for(int k = 0; k < iter_count+1; ++k){
        double tmp_2_norm = euclidean_vec_norm_cpu(&V[k*n_rows], n_rows);
        if(std::abs(tmp_2_norm) > 1+tol){
            printf("GMRES WARNING: basis vector v_%i has a norm of %.17g, \n \
                    and does not have a norm of 1.0 as was expected.\n", k, tmp_2_norm);
        }
        else{
            for(int j = iter_count; j > 0; --j){
                double tmp_dot;
                // Takes new v_k, and compares with all other basis vectors in V
                dot(&V[(iter_count+1)*n_rows], &V[j*n_rows], &tmp_dot, n_rows);
                if(std::abs(tmp_dot) > tol){
                    printf("GMRES WARNING: basis vector v_%i is not orthogonal to basis vector v_%i, \n \
                            their dot product is %.17g, and not 0.0 as was expected.\n", k, j, tmp_dot);
                }
            }
        }
    }
#endif
#endif

    timers->gmres_orthog_wtime->end_stopwatch();

    ///////////////////// Least Squares Step /////////////////////

    timers->gmres_leastsq_wtime->start_stopwatch();

    // Per-iteration "local" Givens rotation (m+1 x m+1) matrix
    init_identity<VT>(J, 0.0, (restart_len+1), (restart_len+1)); 

    // The effect all of rotations so far upon the (k+1 x k) Hesseberg matrix
    // (except we store as row vectors, for pointer reasons...)
    init_identity<VT>(H_tmp, 0.0, (restart_len+1), restart_len); 

#ifdef DEBUG_MODE
    std::cout << "H_tmp_old" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(H_tmp[(restart_len*row_idx) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif
#ifdef FINE_TIMERS
    timers->gmres_compute_H_tmp_wtime->start_stopwatch();
#endif
    if(iter_count == 0){
        // Just copies H
        // TODO: just need to copy first column
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                H_tmp[restart_len*row_idx + col_idx] = H[restart_len*row_idx + col_idx];
            }
        }
    }
    else{
        // Compute H_tmp = Q*H (dense MMM) (m+1 x m) = (m+1 x m+1)(m+1 x m)(i.e. perform all rotations on H)
        // NOTE: Could cut the indices in half+1, since only an "upper" (lower here) hessenberg matrix mult
        dense_MMM_t_t<VT>(Q, H, H_tmp, (restart_len+1), (restart_len+1), restart_len);
    }
#ifdef FINE_TIMERS
    timers->gmres_compute_H_tmp_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
    std::cout << "H_tmp_new" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(H_tmp[(restart_len*row_idx) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Form Givens rotation matrix for next iteration
    // NOTE: since J is typically column accessed, need to transpose to access rows
    double J_denom = std::sqrt(std::pow(static_cast<double>(H_tmp[(iter_count*restart_len) + iter_count]),2) + \
                     std::pow(static_cast<double>(H_tmp[(iter_count+1)*restart_len + iter_count]),2));

    double c_i = H_tmp[(iter_count*restart_len) + iter_count] / J_denom;
    double s_i = H_tmp[((iter_count+1)*restart_len) + iter_count] / J_denom;

    // J[0][0] locally
    J[iter_count*(restart_len+1) + iter_count] = c_i;
    // J[0][1] locally
    J[iter_count*(restart_len+1) + (iter_count+1)] = s_i;
    // J[1][0] locally
    J[(iter_count+1)*(restart_len+1) + iter_count] = -1.0 * s_i;
    // J[1][1] locally
    J[(iter_count+1)*(restart_len+1) + (iter_count+1)] = c_i;

#ifdef DEBUG_MODE
    std::cout << "J" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(J[((restart_len+1)*row_idx) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

#ifdef DEBUG_MODE
    std::cout << "old_Q" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(Q[(restart_len+1)*row_idx + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Combine local Givens rotations with all previous, 
    // i.e. compute Q <- J*Q (dense MMM)
#ifdef FINE_TIMERS
    timers->gmres_compute_Q_wtime->start_stopwatch();
#endif
    dense_MMM_t_t(J, Q, Q_copy, (restart_len+1), (restart_len+1), (restart_len+1));

    // std::swap(Q_copy, Q);
    // Q = Q_copy;
    // ^ TODO: lazy copy
    for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
        for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
            Q[(restart_len+1)*row_idx + col_idx] = Q_copy[(restart_len+1)*row_idx + col_idx];
        }
    }
#ifdef FINE_TIMERS
    timers->gmres_compute_Q_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
    std::cout << "new_Q" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(Q[(restart_len+1)*row_idx + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // R <- Q*H (dense MMM) (m+1 x m) <- (m+1 x m+1)(m+1 x m)
#ifdef FINE_TIMERS
    timers->gmres_compute_R_wtime->start_stopwatch();
#endif
    dense_MMM_t_t(Q, H, R, (restart_len+1), (restart_len+1), restart_len);
#ifdef FINE_TIMERS
    timers->gmres_compute_R_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
    std::cout << "R" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(R[(row_idx*restart_len) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

#ifdef DEBUG_MODE
    // Sanity check: Validate that H == Q_tR ((m+1 x m) == (m+1 x m+1)(m+1 x m))

    VT *Q_t = new VT[(restart_len+1) * (restart_len+1)];
    init<VT>(Q_t, 0.0, (restart_len+1) * (restart_len+1));

    dense_transpose<VT>(Q, Q_t, restart_len+1, restart_len+1);

    std::cout << "Q_t" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(Q_t[(restart_len+1)*row_idx + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    VT *Q_tR = new VT[(restart_len+1) * (restart_len)];
    init<VT>(Q_tR, 0.0, (restart_len+1) * (restart_len));

    // Compute Q_tR (dense MMM) (m+1 x m) <- (m+1 x m+1)(m+1 x m)
    for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
        for(int col_idx = 0; col_idx < restart_len; ++col_idx){
            VT tmp = 0.0;
            strided_2_dot<VT>(&Q_t[row_idx*(restart_len+1)], &R[col_idx], &tmp, restart_len+1, restart_len);
            Q_tR[(row_idx*restart_len) + col_idx] = tmp;
        }
    }

    // Print Q_tR
    std::cout << "Q_tR" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << static_cast<double>(Q_tR[(row_idx*restart_len) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    // Scan and validate H=Q_tR
    for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
        for(int col_idx = 0; col_idx < restart_len; ++col_idx){
            int idx = row_idx*restart_len + col_idx;
            if(std::abs(static_cast<double>(Q_tR[idx] - H[idx])) > tol){
                printf("GMRES ERROR: The Q_tR factorization of H at index %i has a value %.17g, \n \
                    and does not have a value of %.17g as was expected.\n", \
                    row_idx*restart_len + col_idx, Q_tR[idx], H[row_idx*restart_len + col_idx]);
            }
        }
    }
#endif

#ifdef DEBUG_MODE
    std::cout << "g_" << iter_count << " = [\n";
    for(int i = 0; i <= restart_len; ++i){
        std::cout << static_cast<double>(g[i])  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // g_k+1 <- Q* g_k (dMVM) ((m+1 x 1) = (m+1 x m+1)(m+1 x 1))
    init<VT>(g_copy, 0.0, restart_len+1);
    g_copy[0] = beta;
    init<VT>(g, 0.0, restart_len+1);
    g[0] = beta;
    dense_MMM<VT>(Q, g, g_copy, (restart_len+1), (restart_len+1), 1);

    // TODO: lazy copy
    for(int row_idx = 0; row_idx < restart_len+1; ++row_idx){
        g[row_idx] = g_copy[row_idx];
    }

#ifdef DEBUG_MODE
    std::cout << "g_" << iter_count+1 << " = [\n";
    for(int i = 0; i <= restart_len; ++i){
        std::cout << static_cast<double>(g[i])  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // Extract the last element from g as residual norm
    *residual_norm = std::abs(static_cast<double>(g[iter_count + 1]));

#ifdef DEBUG_MODE
    std::cout << "residual_norm = " << static_cast<double>(*residual_norm) << std::endl;
#endif


#ifdef DEBUG_MODE
    delete Q_tR;
#endif

    timers->gmres_leastsq_wtime->end_stopwatch();
}

template <typename VT>
void allocate_gmres_structs(
    gmresArgs<VT> *gmres_args,
    int vec_size
){
    std::cout << "Allocating space for GMRES structs" << std::endl;
    VT *init_v = new VT[vec_size];
    VT *V = new VT[vec_size * (gmres_args->restart_length + 1)]; // (m x n)
    VT *Vy = new VT[vec_size]; // (m x 1)
    VT *H = new VT[(gmres_args->restart_length + 1) * gmres_args->restart_length]; // (m+1 x m) 
    VT *H_tmp = new VT[(gmres_args->restart_length + 1) * gmres_args->restart_length]; // (m+1 x m)
    VT *J = new VT[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)];
    VT *R = new VT[gmres_args->restart_length * (gmres_args->restart_length + 1)]; // (m+1 x m)
    VT *Q = new VT[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)]; // (m+1 x m+1)
    VT *Q_copy = new VT[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)]; // (m+1 x m+1)
    VT *g = new VT[gmres_args->restart_length + 1];
    VT *g_copy = new VT[gmres_args->restart_length + 1];

#ifdef USE_AP
    double *V_dp = new double[vec_size * (gmres_args->restart_length + 1)];
    double *Vy_dp = new double[vec_size];
    float *V_sp = new float[vec_size * (gmres_args->restart_length + 1)];
    float *Vy_sp = new float[vec_size];
#ifdef HAVE_HALF_MATH
    _Float16 *V_hp = new _Float16[vec_size * (gmres_args->restart_length + 1)];
    _Float16 *Vy_hp = new _Float16[vec_size];
#endif
#endif

    gmres_args->init_v = init_v;
    gmres_args->V = V;
    gmres_args->Vy = Vy;
    gmres_args->H = H;
    gmres_args->H_tmp = H_tmp;
    gmres_args->J = J;
    gmres_args->R = R;
    gmres_args->Q = Q;
    gmres_args->Q_copy = Q_copy;
    gmres_args->g = g;
    gmres_args->g_copy = g_copy;
    gmres_args->restart_count = 0;

#ifdef USE_AP
    gmres_args->V_dp = V_dp;
    gmres_args->Vy_dp = Vy_dp;
    gmres_args->V_sp = V_sp;
    gmres_args->Vy_sp = Vy_sp;
#ifdef HAVE_HALF_MATH
    gmres_args->V_hp = V_hp;
    gmres_args->Vy_hp = Vy_hp;
#endif
#endif
}

template <typename VT>
void init_gmres_structs(
    gmresArgs<VT> *gmres_args,
    VT *r,
    int n_rows
){
    int restart_len = gmres_args->restart_length;

    gmres_args->beta = euclidean_vec_norm_cpu(r, n_rows);

    scale_residual<VT>(gmres_args->init_v, r, 1 / gmres_args->beta, n_rows);

#ifdef DEBUG_MODE
    std::cout << "Beta = " << gmres_args->beta << std::endl;
    std::cout << "init_v = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << static_cast<double>(gmres_args->init_v[i]) << ", ";
        }
    std::cout << "]" << std::endl;
#endif
    
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double"){
        init<double>(gmres_args->V_dp, 0.0, n_rows * (restart_len+1));
        init<double>(gmres_args->Vy_dp, 0.0, n_rows);
    }
    else if(xstr(WORKING_PRECISION) == "float"){
        init<float>(gmres_args->V_sp, 0.0f, n_rows * (restart_len+1));
        init<float>(gmres_args->Vy_sp, 0.0f, n_rows);
    }
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        init<_Float16>(gmres_args->V_hp, 0.0f16, n_rows * (restart_len+1));
        init<_Float16>(gmres_args->Vy_hp, 0.0f16, n_rows);
#endif
    }
#else
    init<VT>(gmres_args->V, 0.0, n_rows * (restart_len+1));
    init<VT>(gmres_args->Vy, 0.0, n_rows);
#endif

    // Give v0 to first row of V
    #pragma omp parallel for
    for(int i = 0; i < n_rows; ++i){
        gmres_args->V[i] = gmres_args->init_v[i];
    }
    
    init<VT>(gmres_args->H, 0.0, restart_len * (restart_len+1));

    init_identity<VT>(gmres_args->R, 0.0, restart_len, (restart_len+1));
    
    init_identity<VT>(gmres_args->Q, 0.0, (restart_len+1), (restart_len+1));

    init_identity<VT>(gmres_args->Q_copy, 0.0, (restart_len+1), (restart_len+1));

    init<VT>(gmres_args->g, 0.0, restart_len+1);
    gmres_args->g[0] = gmres_args->beta; // <- supply starting element
    
    init<VT>(gmres_args->g_copy, 0.0, restart_len+1);
    gmres_args->g_copy[0] = gmres_args->beta; // <- supply starting element
}

void init_gmres_timers(Timers *timers){
    timeval *gmres_spmv_start = new timeval;
    timeval *gmres_spmv_end = new timeval;
    Stopwatch *gmres_spmv_wtime = new Stopwatch(gmres_spmv_start, gmres_spmv_end);
    timers->gmres_spmv_wtime = gmres_spmv_wtime;

    timeval *gmres_orthog_start = new timeval;
    timeval *gmres_orthog_end = new timeval;
    Stopwatch *gmres_orthog_wtime = new Stopwatch(gmres_orthog_start, gmres_orthog_end);
    timers->gmres_orthog_wtime = gmres_orthog_wtime;

    timeval *gmres_mgs_start = new timeval;
    timeval *gmres_mgs_end = new timeval;
    Stopwatch *gmres_mgs_wtime = new Stopwatch(gmres_mgs_start, gmres_mgs_end);
    timers->gmres_mgs_wtime = gmres_mgs_wtime;

    timeval *gmres_mgs_dot_start = new timeval;
    timeval *gmres_mgs_dot_end = new timeval;
    Stopwatch *gmres_mgs_dot_wtime = new Stopwatch(gmres_mgs_dot_start, gmres_mgs_dot_end);
    timers->gmres_mgs_dot_wtime = gmres_mgs_dot_wtime;

    timeval *gmres_mgs_sub_start = new timeval;
    timeval *gmres_mgs_sub_end = new timeval;
    Stopwatch *gmres_mgs_sub_wtime = new Stopwatch(gmres_mgs_sub_start, gmres_mgs_sub_end);
    timers->gmres_mgs_sub_wtime = gmres_mgs_sub_wtime;

    timeval *gmres_leastsq_start = new timeval;
    timeval *gmres_leastsq_end = new timeval;
    Stopwatch *gmres_leastsq_wtime = new Stopwatch(gmres_leastsq_start, gmres_leastsq_end);
    timers->gmres_leastsq_wtime = gmres_leastsq_wtime;

    timeval *gmres_compute_H_tmp_start = new timeval;
    timeval *gmres_compute_H_tmp_end = new timeval;
    Stopwatch *gmres_compute_H_tmp_wtime = new Stopwatch(gmres_compute_H_tmp_start, gmres_compute_H_tmp_end);
    timers->gmres_compute_H_tmp_wtime = gmres_compute_H_tmp_wtime;

    timeval *gmres_compute_Q_start = new timeval;
    timeval *gmres_compute_Q_end = new timeval;
    Stopwatch *gmres_compute_Q_wtime = new Stopwatch(gmres_compute_Q_start, gmres_compute_Q_end);
    timers->gmres_compute_Q_wtime = gmres_compute_Q_wtime;

    timeval *gmres_compute_R_start = new timeval;
    timeval *gmres_compute_R_end = new timeval;
    Stopwatch *gmres_compute_R_wtime = new Stopwatch(gmres_compute_R_start, gmres_compute_R_end);
    timers->gmres_compute_R_wtime = gmres_compute_R_wtime;

    timeval *gmres_get_x_start = new timeval;
    timeval *gmres_get_x_end = new timeval;
    Stopwatch *gmres_get_x_wtime = new Stopwatch(gmres_get_x_start, gmres_get_x_end);
    timers->gmres_get_x_wtime = gmres_get_x_wtime;

    timeval *gmres_apply_preconditioner_start = new timeval;
    timeval *gmres_apply_preconditioner_end = new timeval;
    Stopwatch *gmres_apply_preconditioner_wtime = new Stopwatch(gmres_apply_preconditioner_start, gmres_apply_preconditioner_end);
    timers->gmres_apply_preconditioner_wtime = gmres_apply_preconditioner_wtime;
}

template <typename VT>
void gmres_get_x(
    VT *R,
    VT *g,
    VT *x,
    VT *x_0,
    VT *V,
    VT *Vy,
#ifdef USE_AP
    double *x_dp,
    double *x_0_dp,
    double *V_dp,
    double *Vy_dp,
    float *x_sp,
    float *x_0_sp,
    float *V_sp,
    float *Vy_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *x_hp,
    _Float16 *x_0_hp,
    _Float16 *V_hp,
    _Float16 *Vy_hp,
#endif
#endif
    int n_rows,
    int restart_count,
    int iter_count,
    int restart_len
){
    std::vector<VT> y(restart_len, 0.0);

    double diag_elem = 0.0;
    double sum;

    // Adjust for restarting
    iter_count -= restart_count* restart_len;

#ifdef DEBUG_MODE
    std::cout << "gmres_get_x iter_count = " << iter_count << std::endl;
    std::cout << "gmres_get_x restart_count = " << restart_count << std::endl;
#endif

#ifdef DEBUG_MODE
    std::cout << "when solving for x, R" << " = [\n";
    for(int row_idx = iter_count; row_idx >= 0; --row_idx){
        for(int col_idx = iter_count; col_idx >= 0; --col_idx){
                std::cout << std::setw(11);
                std::cout << static_cast<double>(R[(row_idx*restart_len) + col_idx])  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // (dense) Backward triangular solve Ry = g ((m+1 x m)(m x 1) = (m+1 x 1))
    // Traverse R \in \mathbb{R}^(m+1 x m) from last to first row
    for(int row_idx = iter_count; row_idx >= 0; --row_idx){
        sum = 0.0;
        for(int col_idx = row_idx; col_idx < restart_len; ++col_idx){
            if(row_idx == col_idx){
                diag_elem = R[(row_idx*restart_len) + col_idx];
            }
            else{
                sum += R[(row_idx*restart_len) + col_idx] * y[col_idx];
            }
            
        }
        y[row_idx] = (g[row_idx] - sum) / diag_elem;
#ifdef DEBUG_MODE_FINE
        std::cout << g[row_idx] << " - " << sum << " / " << diag_elem << std::endl; 
#endif
    }

#ifdef DEBUG_MODE
    std::cout << "y_" << iter_count << " = [\n";
    for(int i = 0; i < restart_len; ++i){
        std::cout << static_cast<double>(y[i])  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // (dense) matrix vector multiply Vy <- V*y ((n x 1) = (n x m)(m x 1))
    // dense_MMM_t<VT>(V, &y[0], Vy, n_rows, restart_len, 1);

#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double")
        dense_MMM_t<double>(V_dp, &y_dp[0], Vy_dp, n_rows, restart_len, 1);
    else if(xstr(WORKING_PRECISION) == "float")
        dense_MMM_t<float>(V_sp, &y_sp[0], Vy_sp, n_rows, restart_len, 1);
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        dense_MMM_t<_Float16>(V_hp, &y_hp[0], Vy_hp, n_rows, restart_len, 1);
#endif
    }
#else
    dense_MMM_t<VT>(V, &y[0], Vy, n_rows, restart_len, 1);
#endif

#ifdef DEBUG_MODE
    std::cout << "Vy_" << iter_count << " = [\n";
    for(int i = 0; i < n_rows; ++i){
        std::cout << static_cast<double>(Vy[i])  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // // Finally, solve for x ((n x 1) = (n x 1) + (n x m)(m x 1))
#ifdef USE_AP
    if(xstr(WORKING_PRECISION) == "double"){
        for(int i = 0; i < n_rows; ++i){
            x_dp[i] = x_0_dp[i] + Vy_dp[i];
        }
    }
    else if(xstr(WORKING_PRECISION) == "float"){
        for(int i = 0; i < n_rows; ++i){
            x_sp[i] = x_0_sp[i] + Vy_sp[i];
        }
    }   
    else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
        for(int i = 0; i < n_rows; ++i){
            x_hp[i] = x_0_hp[i] + Vy_hp[i];
        }
#endif
    }
#else
    for(int i = 0; i < n_rows; ++i){
        x[i] = x_0[i] + Vy[i];
#ifdef DEBUG_MODE_FINE
        std::cout << "x[" << i << "] = " << x_0[i] << " + " << Vy[i] << " = " << x[i] << std::endl; 
#endif
    }
#endif
}
#endif