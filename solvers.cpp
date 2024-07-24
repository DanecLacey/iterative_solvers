#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#include <iomanip>
#include <cmath>

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
        // std::cout << "x_new[" << row_idx << "] = " << b[row_idx] << " - " << sum << " / " <<  diag_elem <<std::endl;
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

void gm_iteration_ref_cpu(
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

void solve_cpu(
    argType *args
){
    std::cout << "Entering Solver Harness" << std::endl;

    double *x = new double[args->vec_size];
    double *x_new = new double[args->vec_size];
    double *x_old = new double[args->vec_size];
    SparseMtxFormat *sparse_mat = args->sparse_mat;
    Flags *flags = args->flags;
    double residual_norm;

    // Need deep copy TODO: rethink and do better
    #pragma omp parallel for
    for(int i = 0; i < args->vec_size; ++i){
        x[i] = args->x_old[i];
        x_new[i] = args->x_new[i];
        x_old[i] = args->x_old[i];
    }

    if(args->flags->print_iters){
        iter_output(x, args->vec_size, args->loop_params->iter_count);
        printf("\n");
    }

    if(args->solver_type == "gmres"){
        init_gmres_structs(args, args->vec_size);
        init_gmres_timers(args);
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << x[i] << std::endl;
    }
#endif

#ifdef USE_LIKWID
    #pragma omp parallel
    {
#ifdef USE_USPMV
#ifdef USE_AP
        LIKWID_MARKER_REGISTER("uspmv_ap_crs_benchmark");
#else
        LIKWID_MARKER_REGISTER("uspmv_crs_benchmark");
#endif

#else 
        LIKWID_MARKER_REGISTER("native_spmv_benchmark");
#endif
    }
#endif


    do{
#ifdef DEBUG_MODE
        std::cout << "Restarted GMRES inputting the x vector [" << std::endl;
        for(int i = 0; i < args->vec_size; ++i){
            std::cout << x[i] << std::endl;
        }
        std::cout << "]" << std::endl;
#endif

        args->timers->solver_wtime->start_stopwatch();
        // TODO: change to solver class methods
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            jacobi_iteration_ref_cpu(
                args->sparse_mat, 
                args->D, 
                args->b, 
                x_old, 
                x_new
            );
            // jacobi_iteration_sep_cpu(sparse_mat, D, b, x_old, x_new, args->vec_size);
        }
        else if(args->solver_type == "gauss-seidel"){
            // For a reference solution, not meant for use with USpMV library
            gs_iteration_ref_cpu(
                args->sparse_mat, 
                args->tmp, 
                args->D, 
                args->b, 
                x
            );
            // gs_iteration_sep_cpu(sparse_mat, tmp, D, b, x, args->vec_size);
        }
        else if(args->solver_type == "gmres"){
            gm_iteration_ref_cpu(
                sparse_mat, 
                args->timers,
                args->V,
                args->H,
                args->H_tmp,
                args->J,
                args->Q,
                args->Q_copy,
                args->tmp, 
                args->R,
                args->g,
                args->g_copy,
                args->b, 
                x,
                args->beta,
                args->vec_size,
                args->restart_count,
                args->loop_params->iter_count,
                &residual_norm,
                args->loop_params->gmres_restart_len
            );
        }
        args->timers->solver_wtime->end_stopwatch();

        // Record residual every "residual_check_len" iterations
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            // if(args->solver_type == "gmres"){
            //     args->timers->gmres_get_x_wtime->start_stopwatch();
            //     gmres_get_x(args->R, args->g, x, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count, args->loop_params->gmres_restart_len);
            //     args->timers->gmres_get_x_wtime->end_stopwatch();
            // }
            record_residual_norm(args, flags, sparse_mat, &residual_norm, args->r, x, args->b, x_new, args->tmp);
        }

        if(flags->print_iters)
            print_x(args, x, x_new, x_old, args->vec_size);  

        // Need to swap arrays every iteration in the case of Jacobi solver
        if(args->solver_type == "jacobi")
            std::swap(x_new, x_old);

#ifdef DEBUG_MODE
        std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
#endif 
        if(args->solver_type == "gmres"){
            // Decide to restart or not 
            if(residual_norm <= args->loop_params->stopping_criteria || \
                args->loop_params->iter_count >= args->loop_params->max_iters){
                    
            }
            else if ( (args->loop_params->iter_count+1) % args->gmres_restart_len == 0 ){
                // Restart GMRES
#ifdef DEBUG_MODE
                std::cout << "RESTART GMRES" << std::endl;
#endif
                args->timers->gmres_get_x_wtime->start_stopwatch();
                gmres_get_x(args->R, args->g, x, x_old, args->V, args->Vy, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count, args->loop_params->gmres_restart_len);
                args->timers->gmres_get_x_wtime->end_stopwatch();
                calc_residual_cpu(args->sparse_mat, x, args->b, args->r, args->tmp, args->coo_mat->n_cols);
#ifdef DEBUG_MODE
                printf("restart residual = [");
                for(int i = 0; i < args->coo_mat->n_cols; ++i){
                    std::cout << args->r[i] << ",";
                }
                printf("]\n");
#endif
                args->beta = euclidean_vec_norm_cpu(args->r, args->coo_mat->n_cols); 
                scale(args->init_v, args->r, 1 / args->beta, args->coo_mat->n_cols);
    
#ifdef DEBUG_MODE
                std::cout << "Restarted Beta = " << args->beta << std::endl;          

                std::cout << "init_v = [";
                    for(int i = 0; i < args->vec_size; ++i){
                        std::cout << args->init_v[i] << ", ";
                    }
                std::cout << "]" << std::endl;
#endif
                double norm_r0 = euclidean_vec_norm_cpu(args->r, args->vec_size);

#ifdef DEBUG_MODE
                printf("restarted norm(initial residual) = %f\n", norm_r0);
#endif
                init_gmres_structs(args, args->vec_size);
                ++args->restart_count;
#ifdef DEBUG_MODE
                std::cout << "Restarted GMRES outputting the x vector [" << std::endl;
                for(int i = 0; i < args->vec_size; ++i){
                    std::cout << x[i] << std::endl;
                }
                std::cout << "]" << std::endl;
#endif

                // Need deep copy TODO: rethink and do better
                #pragma omp parallel for
                for(int i = 0; i < args->vec_size; ++i){
                    x_old[i] = x[i];
                }

            }
        }

        ++args->loop_params->iter_count;


        // std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;

    } while(residual_norm > args->loop_params->stopping_criteria && \
    args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    if(args->solver_type == "jacobi"){
        // TODO: Bandaid, sincne swapping seems to not work :(
        #pragma omp parallel for
        for(int i = 0; i < args->vec_size; ++i){
            args->x_star[i] = x_old[i];
            // std::cout << "args->x_star[" << i << "] = " << x_old[i] << "==" << args->x_star[i] << std::endl;
        }
        // std::swap(args->x_star, x_old);

    }
    else if (args->solver_type == "gauss-seidel"){
        #pragma omp parallel for
        for(int i = 0; i < args->vec_size; ++i){
            args->x_star[i] = x[i];
            // std::cout << "args->x_star[" << i << "] = " << x[i] << "==" << args->x_star[i] << std::endl;
        }
        // std::swap(x, args->x_star);
    }    
    
    // Record final residual with approximated solution vector x_star
    if(args->solver_type == "gmres"){
        // Only needed if you actually want the x vector
        // args->timers->gmres_get_x_wtime->start_stopwatch();
        // gmres_get_x(args->R, args->g, args->x_star, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count - 1, args->loop_params->gmres_restart_len);
        // args->timers->gmres_get_x_wtime->end_stopwatch();
    }

    record_residual_norm(args, flags, sparse_mat, &residual_norm, args->r, args->x_star, args->b, args->x_star, args->tmp);

#ifdef USE_USPMV
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[args->vec_size];
    apply_permutation(x_star_perm, args->x_star, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);

    // Deep copy, so you can free memory
    // NOTE: You do not take SCS padding with you! 
    #pragma omp parallel for
    for(int i = 0; i < args->coo_mat->n_cols; ++i){
        args->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
#endif

    delete x;
    delete x_new;
    delete x_old;
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

void solve_gpu(
    argType *args
){

    // NOTE: Only for convenience. Will change to UM later.
    double *h_residual_norm = new double;

    // TODO: Why does this get messed up? 
    args->loop_params->residual_count = 0;

    // // Unpack relevant args
    // double *d_x = args->d_x_old; // GS
    // double *d_x_new = args->d_x_new; // Jacobi
    // double *d_x_old = args->d_x_old; // Jacobi
    // int d_n_rows = args->coo_mat->n_rows;

    // // TODO: collect into a struct
    // int *d_row_ptr = args->d_row_ptr;
    // int *d_col = args->d_col;
    // double *d_val = args->d_val; 

    // double *d_tmp = args->d_tmp;
    // double *d_D = args->d_D;
    // double *d_r = args->d_r;
    // double *d_b = args->d_b;

    double *d_residual_norm;
    cudaMalloc(&d_residual_norm, sizeof(double));
    cudaMemset(d_residual_norm, 0.0, sizeof(double));

    Flags *flags = args->flags; 

    double residual_norm;

    // TODO: Adapt for GPUs
    // if(args->flags->print_iters){
    //     iter_output(d_x, args->loop_params->iter_count);
    //     printf("\n");
    // }

    // TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//     std::cout << "x vector:" << std::endl;
//     for(int i = 0; i < args->vec_size; ++i){
//         std::cout << d_x[i] << std::endl;
//     }
// #endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            // jacobi_iteration_ref_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new, args->vec_size);
            jacobi_iteration_sep_gpu(args->vec_size, args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new);
        }
        else if(args->solver_type == "gauss-seidel"){
            // TODO: Adapt for GPUs
            printf("GS_solve still under development for GPUs.\n");
            exit(1);
            // For a reference solution, not meant for use with USpMV library
            // gs_iteration_ref_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
            // gs_iteration_sep_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
        }
        
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            if(args->solver_type == "jacobi"){
                calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_new, args->d_b, args->d_r, args->d_tmp, args->vec_size);
            }
            else if(args->solver_type == "gauss-seidel"){
                // TODO: Adapt for GPUs
                printf("GS_solve still under development for GPUs.\n");
                exit(1);
                // calc_residual_gpu(sparse_mat, x, b, r, tmp);
            }
            
///////////////////////////////////// Grrr DEBUG! //////////////////////////////////////////
            // For now, have to do this on the CPU. Giving up on GPU implementation
            // cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
            // *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);
            // TODO: Correct grid + block size?
            // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
            // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
            infty_vec_norm_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(double)>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////// DEBUG! //////////////////////////////////////////

            // TODO: Put residual_norm in unified memory to avoid this transfer
            // NOTE: need to convert *double to *void
            cudaMemcpy(&(*h_residual_norm), &(*d_residual_norm), sizeof(double), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize(); // <- not necessary
            // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

            // std::cout << "the first h_residual_norm = " << *h_residual_norm << std::endl;
            // exit(0);
            
            args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;
            ++args->loop_params->residual_count;

// TODO: Adapt for GPUs
//             if(flags->print_iters){
//                 if(args->solver_type == "jacobi"){
//                     iter_output(x_new, args->loop_params->iter_count);
//                 }
//                 else if(args->solver_type == "gauss-seidel"){
//                     iter_output(x, args->loop_params->iter_count);
//                 }
//             }
        }

// TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//         std::cout << "[";
//         if(args->solver_type == "jacobi"){
//             for(int i = 0; i < x_new->size(); ++i){
//                 std::cout << (*x_new)[i] << ", ";
//             }
//         }
//         else if (args->solver_type == "gauss-seidel"){
//             for(int i = 0; i < x->size(); ++i){
//                 std::cout << (*x)[i] << ", ";
//             }
//         }
//         std::cout << "]" << std::endl;
  
//         std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
//         std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
// #endif  

// TODO: Adapt for GPUs???
        cudaDeviceSynchronize();
        if(args->solver_type == "jacobi"){
            // NOTE: Might work, might not..
            // TODO: huh?? Causes seg fault
            // std::cout << "d_x_new pointer: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer: " << d_x_old << std::endl;
            std::swap(args->d_x_new, args->d_x_old);
            // std::cout << "d_x_new pointer after swap: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer after swap: " << d_x_old << std::endl;
        }
    
        ++args->loop_params->iter_count;

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    // Do check on host for now, easiest
        // std::cout << *h_residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
        // exit(0);
    } while(*h_residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (*h_residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    cudaDeviceSynchronize();
    if(args->solver_type == "jacobi"){
        // TODO: huh?? Causes seg fault
        std::swap(args->d_x_old, args->d_x_star);
    }
    else if (args->solver_type == "gauss-seidel"){
        // TODO: Adapt for GPUs
        printf("GS_solve still under development for GPUs.\n");
        exit(1);
        // std::swap(*x, *(args->x_star));
    }

    // Record final residual with approximated solution vector x
///////////////////////////////////// DEBUG! //////////////////////////////////////////
    // TODO: Giving up on GPU for this for now
    cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
    *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);

    // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
    // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////////////////////////////////////////////////////////

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;

// TODO: Adapt for GPUs
// #ifdef USE_USPMV
//     // Bring final result vector out of permuted space
//     std::vector<double> x_star_perm(args->vec_size, 0);
//     apply_permutation(&(x_star_perm)[0], &(*args->x_star)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
//     std::swap(x_star_perm, (*args->x_star));
// #endif

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);

    // Why are you freeing this here?
    cudaFree(d_residual_norm);
    delete h_residual_norm;
}
#endif

void solve(
    argType *args
){
    timeval *solver_harness_time_start = new timeval;
    timeval *solver_harness_time_end = new timeval;
    Stopwatch *solver_harness_wtime = new Stopwatch(solver_harness_time_start, solver_harness_time_end);
    args->timers->solver_harness_wtime = solver_harness_wtime ;
    args->timers->solver_harness_wtime->start_stopwatch();

    timeval *solver_time_start = new timeval;
    timeval *solver_time_end = new timeval;
    Stopwatch *solver_wtime = new Stopwatch(solver_time_start, solver_time_end);
    args->timers->solver_wtime = solver_wtime ;

#ifndef __CUDACC__
    solve_cpu(args);
#else
    solve_gpu(args);
#endif

    args->timers->solver_harness_wtime->end_stopwatch();
}