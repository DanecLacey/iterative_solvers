#include <iomanip>
#include <cmath>

#include "../kernels.hpp"
#include "../structs.hpp"
#include "../utility_funcs.hpp"

#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif

void gmres_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    Timers *timers,
    double *V,
    double *H,
    double *H_tmp,
    double *J,
    double *Q,
    double *Q_copy,
    double *w,
    double *R,
    double *g,
    double *g_copy,
    double *b,
    double *x,
    double beta,
    int n_rows,
    int restart_count,
    int iter_count,
    double *residual_norm,
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
    timers->gmres_spmv_wtime->start_stopwatch();
#ifdef USE_USPMV

#ifdef USE_AP

    uspmv_omp_csr_ap_cpu<int>(
        sparse_mat->scs_mat_hp->n_chunks,
        sparse_mat->scs_mat_hp->C,
        &(sparse_mat->scs_mat_hp->chunk_ptrs)[0],
        &(sparse_mat->scs_mat_hp->chunk_lengths)[0],
        &(sparse_mat->scs_mat_hp->col_idxs)[0],
        &(sparse_mat->scs_mat_hp->values)[0],
        &V[iter_count*n_rows],
        w,
        sparse_mat->scs_mat_lp->n_chunks,
        sparse_mat->scs_mat_lp->C,
        &(sparse_mat->scs_mat_lp->chunk_ptrs)[0],
        &(sparse_mat->scs_mat_lp->chunk_lengths)[0],
        &(sparse_mat->scs_mat_lp->col_idxs)[0],
        &(sparse_mat->scs_mat_lp->values)[0]
    );

#else
    uspmv_omp_csr_cpu(
        sparse_mat->scs_mat->C,
        sparse_mat->scs_mat->n_chunks,
        &(sparse_mat->scs_mat->chunk_ptrs)[0],
        &(sparse_mat->scs_mat->chunk_lengths)[0],
        &(sparse_mat->scs_mat->col_idxs)[0],
        &(sparse_mat->scs_mat->values)[0],
        &V[iter_count*n_rows],
        w
    );
#endif

#else
    spmv_crs_cpu(w, sparse_mat->crs_mat, &V[iter_count*n_rows]);
#endif

    timers->gmres_spmv_wtime->end_stopwatch();

#ifdef DEBUG_MODE
    std::cout << "w = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << w[i] << ", ";
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
        dot(w, &V[j*n_rows], &H[iter_count + j*restart_len] , n_rows);
#ifdef FINE_TIMERS
        timers->gmres_mgs_dot_wtime->end_stopwatch(); 
#endif

#ifdef DEBUG_MODE
        std::cout << "h_" << j << "_" << iter_count << " = " << H[iter_count + j*restart_len] << std::endl;
#endif

        // w_j <- w_j - h_ij*v_k
#ifdef FINE_TIMERS
        timers->gmres_mgs_sub_wtime->start_stopwatch();
#endif
        subtract_vectors_cpu(w, w, &V[j*n_rows], n_rows, H[iter_count + j*restart_len]); 
#ifdef FINE_TIMERS
        timers->gmres_mgs_sub_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
        std::cout << "adjusted_w_" << j << "_rev  = [";
            for(int i = 0; i < n_rows; ++i){
                std::cout << w[i] << ", ";
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
                std::cout << V[(n_rows*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Save norm to Hessenberg matrix subdiagonal H[k+1,k]
    H[(iter_count+1)*restart_len + iter_count] = euclidean_vec_norm_cpu(w, n_rows);

#ifdef DEBUG_MODE
    std::cout << "H" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H[(restart_len*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Normalize the new orthogonal vector v <- v/H[k+1,k]
    scale(&V[(iter_count+1)*n_rows], w, 1.0/H[(iter_count+1)*restart_len + iter_count], n_rows);

#ifdef DEBUG_MODE
    std::cout << "v_" << iter_count+1 << " = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << std::setw(fixed_width);
            std::cout << V[(iter_count+1)*n_rows + i] << ", ";
        }
    std::cout << "]" << std::endl;
#endif

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

    timers->gmres_orthog_wtime->end_stopwatch();

    ///////////////////// Least Squares Step /////////////////////

    timers->gmres_leastsq_wtime->start_stopwatch();

    // Per-iteration "local" Givens rotation (m+1 x m+1) matrix
    init_identity(J, 0.0, (restart_len+1), (restart_len+1)); 

    // The effect all of rotations so far upon the (k+1 x k) Hesseberg matrix
    // (except we store as row vectors, for pointer reasons...)
    init_identity(H_tmp, 0.0, (restart_len+1), restart_len); 

#ifdef DEBUG_MODE
    std::cout << "H_tmp_old" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H_tmp[(restart_len*row_idx) + col_idx]  << ", ";
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
        dense_MMM_t_t(Q, H, H_tmp, (restart_len+1), (restart_len+1), restart_len);
    }
#ifdef FINE_TIMERS
    timers->gmres_compute_H_tmp_wtime->end_stopwatch();
#endif

#ifdef DEBUG_MODE
    std::cout << "H_tmp_new" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H_tmp[(restart_len*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Form Givens rotation matrix for next iteration
    // NOTE: since J is typically column accessed, need to transpose to access rows
    double J_denom = std::sqrt(std::pow(H_tmp[(iter_count*restart_len) + iter_count],2) + \
                     std::pow(H_tmp[(iter_count+1)*restart_len + iter_count],2));

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
                std::cout << J[((restart_len+1)*row_idx) + col_idx]  << ", ";
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
                std::cout << Q[(restart_len+1)*row_idx + col_idx]  << ", ";
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
                std::cout << Q[(restart_len+1)*row_idx + col_idx]  << ", ";
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
                std::cout << R[(row_idx*restart_len) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

#ifdef DEBUG_MODE
    // Sanity check: Validate that H == Q_tR ((m+1 x m) == (m+1 x m+1)(m+1 x m))

    double *Q_t = new double[(restart_len+1) * (restart_len+1)];
    init(Q_t, 0.0, (restart_len+1) * (restart_len+1));

    dense_transpose(Q, Q_t, restart_len+1, restart_len+1);

    std::cout << "Q_t" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx <= restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q_t[(restart_len+1)*row_idx + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    double *Q_tR = new double[(restart_len+1) * (restart_len)];
    init(Q_tR, 0.0, (restart_len+1) * (restart_len));

    // Compute Q_tR (dense MMM) (m+1 x m) <- (m+1 x m+1)(m+1 x m)
    for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
        for(int col_idx = 0; col_idx < restart_len; ++col_idx){
            double tmp = 0.0;
            strided_2_dot(&Q_t[row_idx*(restart_len+1)], &R[col_idx], &tmp, restart_len+1, restart_len);
            Q_tR[(row_idx*restart_len) + col_idx] = tmp;
        }
    }

    // Print Q_tR
    std::cout << "Q_tR" << " = [\n";
        for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
            for(int col_idx = 0; col_idx < restart_len; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q_tR[(row_idx*restart_len) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    // Scan and validate H=Q_tR
    for(int row_idx = 0; row_idx <= restart_len; ++row_idx){
        for(int col_idx = 0; col_idx < restart_len; ++col_idx){
            int idx = row_idx*restart_len + col_idx;
            if(std::abs(Q_tR[idx] - H[idx]) > tol){
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
        std::cout << g[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // g_k+1 <- Q* g_k (dMVM) ((m+1 x 1) = (m+1 x m+1)(m+1 x 1))
    init(g_copy, 0.0, restart_len+1);
    g_copy[0] = beta;
    init(g, 0.0, restart_len+1);
    g[0] = beta;
    dense_MMM(Q, g, g_copy, (restart_len+1), (restart_len+1), 1);

    // TODO: lazy copy
    for(int row_idx = 0; row_idx < restart_len+1; ++row_idx){
        g[row_idx] = g_copy[row_idx];
    }

#ifdef DEBUG_MODE
    std::cout << "g_" << iter_count+1 << " = [\n";
    for(int i = 0; i <= restart_len; ++i){
        std::cout << g[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // Extract the last element from g as residual norm
    *residual_norm = std::abs(g[iter_count + 1]);

#ifdef DEBUG_MODE
    std::cout << "residual_norm = " << *residual_norm << std::endl;
#endif


#ifdef DEBUG_MODE
    delete Q_tR;
#endif

    timers->gmres_leastsq_wtime->end_stopwatch();
}
