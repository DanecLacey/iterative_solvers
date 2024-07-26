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

struct gmresArgs
{
    double beta;
    double *init_v;
    double *V;
    double *Vy;
    double *H;
    double *H_tmp;
    double *J;
    double *R;
    double *Q;
    double *Q_copy;
    double *g;
    double *g_copy;
    int restart_count;
    int restart_length;
};

template <typename VT>
void gmres_iteration_ref_cpu(
    SparseMtxFormat<VT> *sparse_mat,
    Timers *timers,
    double *V,
    double *H,
    double *H_tmp,
    double *J,
    double *Q,
    double *Q_copy,
    double *w,
    double *w_perm,
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
    // NOTE: We don't time the permutations in the case of USpMV, 
    // as that is an aspect of the implementation and not necessary for Sell-C-Sigma
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
    // uspmv_omp_scs_ap_cpu<int>(
    //     sparse_mat->scs_mat_hp->n_chunks,
    //     sparse_mat->scs_mat_hp->C,
    //     &(sparse_mat->scs_mat_hp->chunk_ptrs)[0],
    //     &(sparse_mat->scs_mat_hp->chunk_lengths)[0],
    //     &(sparse_mat->scs_mat_hp->col_idxs)[0],
    //     &(sparse_mat->scs_mat_hp->values)[0],
    //     &V[iter_count*n_rows],
    //     w_perm,
    //     sparse_mat->scs_mat_lp->n_chunks,
    //     sparse_mat->scs_mat_lp->C,
    //     &(sparse_mat->scs_mat_lp->chunk_ptrs)[0],
    //     &(sparse_mat->scs_mat_lp->chunk_lengths)[0],
    //     &(sparse_mat->scs_mat_lp->col_idxs)[0],
    //     &(sparse_mat->scs_mat_lp->values)[0]
    // );

    timers->gmres_spmv_wtime->end_stopwatch();

    // apply_permutation(w, w_perm, &(sparse_mat->scs_mat_hp->old_to_new_idx)[0], n_rows);
#else
    uspmv_omp_csr_cpu<VT, int>(
        sparse_mat->scs_mat->C,
        sparse_mat->scs_mat->n_chunks,
        &(sparse_mat->scs_mat->chunk_ptrs)[0],
        &(sparse_mat->scs_mat->chunk_lengths)[0],
        &(sparse_mat->scs_mat->col_idxs)[0],
        &(sparse_mat->scs_mat->values)[0],
        &V[iter_count*n_rows],
        w
    );
    // uspmv_omp_scs_cpu<VT, int>(
    //     sparse_mat->scs_mat->C,
    //     sparse_mat->scs_mat->n_chunks,
    //     &(sparse_mat->scs_mat->chunk_ptrs)[0],
    //     &(sparse_mat->scs_mat->chunk_lengths)[0],
    //     &(sparse_mat->scs_mat->col_idxs)[0],
    //     &(sparse_mat->scs_mat->values)[0],
    //     &V[iter_count*n_rows],
    //     w_perm
    // );

    timers->gmres_spmv_wtime->end_stopwatch();

    // apply_permutation(w, w_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_rows);
#endif

#else
    spmv_crs_cpu<VT>(w, sparse_mat->crs_mat, &V[iter_count*n_rows]);

    timers->gmres_spmv_wtime->end_stopwatch();
#endif

    

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

void allocate_gmres_structs(
    gmresArgs *gmres_args,
    int vec_size
){
    std::cout << "Allocating space for GMRES structs" << std::endl;
    double *init_v = new double[vec_size];
    double *V = new double[vec_size * (gmres_args->restart_length + 1)]; // (m x n)
    double *Vy = new double[vec_size]; // (m x 1)
    double *H = new double[(gmres_args->restart_length + 1) * gmres_args->restart_length]; // (m+1 x m) 
    double *H_tmp = new double[(gmres_args->restart_length + 1) * gmres_args->restart_length]; // (m+1 x m)
    double *J = new double[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)];
    double *R = new double[gmres_args->restart_length * (gmres_args->restart_length + 1)]; // (m+1 x m)
    double *Q = new double[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)]; // (m+1 x m+1)
    double *Q_copy = new double[(gmres_args->restart_length + 1) * (gmres_args->restart_length + 1)]; // (m+1 x m+1)
    double *g = new double[gmres_args->restart_length + 1];
    double *g_copy = new double[gmres_args->restart_length + 1];

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
}

void init_gmres_structs(
    gmresArgs *gmres_args,
    double *r,
    int n_rows
){
    int restart_len = gmres_args->restart_length;

    gmres_args->beta = euclidean_vec_norm_cpu(r, n_rows);
    scale(gmres_args->init_v, r, 1 / gmres_args->beta, n_rows);

#ifdef DEBUG_MODE
    std::cout << "Beta = " << gmres_args->beta << std::endl;
    std::cout << "init_v = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << gmres_args->init_v[i] << ", ";
        }
    std::cout << "]" << std::endl;
#endif
    

    init<double>(gmres_args->V, 0.0, n_rows * (restart_len+1));

    // Give v0 to first row of V
    #pragma omp parallel for
    for(int i = 0; i < n_rows; ++i){
        gmres_args->V[i] = gmres_args->init_v[i];
    }
    
    init<double>(gmres_args->H, 0.0, restart_len * (restart_len+1));

    init<double>(gmres_args->Vy, 0.0, n_rows);
    
    init_identity<double>(gmres_args->R, 0.0, restart_len, (restart_len+1));
    
    init_identity<double>(gmres_args->Q, 0.0, (restart_len+1), (restart_len+1));

    init_identity<double>(gmres_args->Q_copy, 0.0, (restart_len+1), (restart_len+1));

    init<double>(gmres_args->g, 0.0, restart_len+1);
    gmres_args->g[0] = gmres_args->beta; // <- supply starting element
    
    init<double>(gmres_args->g_copy, 0.0, restart_len+1);
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
}

void gmres_get_x(
    double *R,
    double *g,
    double *x,
    double *x_0,
    double *V,
    double *Vy,
    int n_rows,
    int restart_count,
    int iter_count,
    int restart_len
){
    std::vector<double> y(restart_len, 0.0);

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
                std::cout << R[(row_idx*restart_len) + col_idx]  << ", ";
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
        std::cout << y[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // (dense) matrix vector multiply Vy <- V*y ((n x 1) = (n x m)(m x 1))
    dense_MMM_t<VT>(V, &y[0], Vy, n_rows, restart_len, 1);

#ifdef DEBUG_MODE
    std::cout << "Vy_" << iter_count << " = [\n";
    for(int i = 0; i < n_rows; ++i){
        std::cout << Vy[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // Finally, solve for x ((n x 1) = (n x 1) + (n x m)(m x 1))
    for(int i = 0; i < n_rows; ++i){
        x[i] = x_0[i] + Vy[i];
        // std::cout << "x[" << i << "] = " << x_0[i] << " + " << Vy[i] << " = " << x[i] << std::endl; 
    }
}
#endif