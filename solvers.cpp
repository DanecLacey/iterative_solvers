#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

#include <iomanip>
#include <cmath>


void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new // treat like y
){
    double diag_elem = 0;
    double sum = 0;

    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < sparse_mat->crs_mat->n_rows; ++row_idx){
        sum = 0;
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == sparse_mat->crs_mat->col[nz_idx]){
                diag_elem = sparse_mat->crs_mat->val[nz_idx];
            }
            else{
                sum += sparse_mat->crs_mat->val[nz_idx] * x_old[sparse_mat->crs_mat->col[nz_idx]];
            }
        }
        x_new[row_idx] = (b[row_idx] - sum) / diag_elem;
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
        uspmv_omp_scs_cpu<double, int>(
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
    double *V,
    double *H,
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
    int iter_count,
    double *residual_norm,
    int max_gmres_iters // <- temporary! only for testing
){
    // NOTES:
    // - for temporary "per iteration" arrays, new vectors are used each iteration.
    //   It is probably better to allocate space during preprocessing, and just use that repeatedly.
    // - The orthonormal vectors in V are stored as row vectors

    double tmp;

#ifdef DEBUG_MODE
    // Tolerance for validation checks
    double tol=1e-14;
    int fixed_width = 12;
#endif

    // double *Q_copy = new double[(max_gmres_iters+1) * (max_gmres_iters+1)]; // (m+1 x m+1)
    // init(Q_copy, 0.0, (max_gmres_iters+1) * (max_gmres_iters+1));

    // // reset Q_copyInitialize 1s in identity matrix
    // for(int i = 0; i <= max_gmres_iters; ++i){
    //     for (int j = 0; j <= max_gmres_iters; ++j){
    //         if(i == j){
    //             Q_copy[i*(max_gmres_iters+1) + j] = 1.0;
    //         }
    //     }
    // }

    ///////////////////// Orthogonalization Step /////////////////////
    // Compute w_k = A*v_k (SpMV)
    for(int row_idx = 0; row_idx < n_rows; ++row_idx){
        tmp = 0.0;
        for(int nz_idx = sparse_mat->crs_mat->row_ptr[row_idx]; nz_idx < sparse_mat->crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            // Selects the k-th row of V (i.e. v_k) to multiply with A
            tmp += sparse_mat->crs_mat->val[nz_idx] * V[iter_count*n_rows + sparse_mat->crs_mat->col[nz_idx]];
        }
        w[row_idx] = tmp;
    }

#ifdef DEBUG_MODE
    std::cout << "w = [";
        for(int i = 0; i < n_rows; ++i){
            std::cout << w[i] << ", ";
        }
    std::cout << "]" << std::endl;
#endif

    // Orthogonalize
    // TODO: improve with MGS
    // NOTE: One column of H filled up at a time
    // I believe this would have to be entered on first pass through to get upper left diagonal element
    // for(int j = 0; j <= iter_count; ++j){
    //     dot(w, &V[j*n_rows], &H[iter_count*max_gmres_iters + j] , n_rows); // h_ij <- (w,v)
    //     subtract_vectors_cpu(w, w, &V[j*n_rows], n_rows, H[iter_count*max_gmres_iters + j]); // w <- w - h_ij*v
    // }

    // TODO: This will be very bad for performance. Need a tranposed version for subtract
    // GS
    for(int j = 0; j <= iter_count; ++j){
        dot(w, &V[j*n_rows], &H[iter_count + j*max_gmres_iters] , n_rows); // h_ij <- (w,v)
        // subtract_vectors_cpu(w, w, &V[j*n_rows], n_rows, H[iter_count*max_gmres_iters + j]); // w <- w - h_ij*v
#ifdef DEBUG_MODE
        std::cout << "h_" << j << "_" << iter_count << " = " << H[iter_count + j*max_gmres_iters] << std::endl;
#endif
        for(int i = 0; i < n_rows; ++i){
            w[i] = w[i] - H[iter_count + j*max_gmres_iters]*V[j*n_rows + i];
        }
#ifdef DEBUG_MODE
        std::cout << "adjusted_w_" << j << "_rev  = [";
            for(int i = 0; i < n_rows; ++i){
                std::cout << w[i] << ", ";
            }
        std::cout << "]" << std::endl;
#endif
    }

    // MGS


// #ifdef DEBUG_MODE
// // computed during the dot product
//     std::cout << "H = [";
//         for(int i = 0; i <= iter_count; ++i){
//             std::cout << H[i] << ", ";
//         }
//     std::cout << "]" << std::endl;
// #endif
#ifdef DEBUG_MODE
    std::cout << "H" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H[(max_gmres_iters*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Save norm to Hessenberg matrix subdiagonal H[k+1,k]
    H[(iter_count+1)*max_gmres_iters + iter_count] = euclidean_vec_norm_cpu(w, n_rows);

#ifdef DEBUG_MODE
    std::cout << "H" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H[(max_gmres_iters*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Normalize the new orthogonal vector v <- v/H[k+1,k]
    scale(&V[(iter_count+1)*n_rows], w, 1.0/H[(iter_count+1)*max_gmres_iters + iter_count], n_rows);

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

// Just check orthog for now
// exit(1);

    ///////////////////// Least Squares Step /////////////////////

    // Per-iteration "local" Givens rotation (m+1 x m+1) matrix
    double *J = new double[(max_gmres_iters+1) * (max_gmres_iters+1)];
    init(J, 0.0, (max_gmres_iters+1) * (max_gmres_iters+1)); 

    // Initialize 1s in identity matrix
    for(int i = 0; i <= max_gmres_iters; ++i){
        for (int j = 0; j <= max_gmres_iters; ++j){
            if(i == j){
                J[i*(max_gmres_iters+1) + j] = 1.0;
            }
        }
    }

    // The effect all of rotations so far upon the (k+1 x k) Hesseberg matrix
    // (except we store as row vectors, for pointer reasons...)
    // TODO: Maybe simpler to just allocate same memory as H for now? Although very unnecessary
    double *H_tmp = new double[(max_gmres_iters+1) * max_gmres_iters]; // (m+1 x m)
    init(H_tmp, 0.0, (max_gmres_iters+1) * max_gmres_iters); 
    
    // Initialize 1s in identity matrix
    for(int i = 0; i <= max_gmres_iters; ++i){
        for (int j = 0; j <= max_gmres_iters; ++j){
            if(i == j){
                H_tmp[i*max_gmres_iters + j] = 1.0;
            }
        }
    }
    
    if(iter_count == 0){
        // Just copies H
        // TODO: just need to copy first column
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                H_tmp[max_gmres_iters*row_idx + col_idx] = H[max_gmres_iters*row_idx + col_idx];
            }
        }
    }
    else{
        
        // Compute H_tmp = Q*H (dense MMM) (m+1 x m) = (m+1 x m+1)(m+1 x m)(i.e. perform all rotations on H)
        // Could cut the indices in half+1, since only an "upper" (lower here) hessenberg matrix mult
        // for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                tmp = 0.0;
                strided_2_dot(&Q[(max_gmres_iters+1)*(row_idx)], &H[col_idx], &tmp, max_gmres_iters+1, max_gmres_iters);
#ifdef DEBUG_MODE
                // Verify subdiagonal is eliminated
                // Toggle for debugging
                // for(int tmp_row_idx = 0; tmp_row_idx <= max_gmres_iters; ++tmp_row_idx){
                //     for(int tmp_col_idx = 0; tmp_col_idx < max_gmres_iters; ++tmp_col_idx){
                //         if(col_idx < row_idx && col_idx < iter_count){
                //             if(std::abs(tmp) > 0.0){
                //                 printf("GMRES WARNING: At index (%i, %i), H_tmp has a value %.17g, \n" \
                //                     "where is was meant to be eliminated! Forcing to 0.0.\n", row_idx, col_idx, tmp);
                //                 tmp = 0.0;
                //             }
                //         }
                //     }
                // }
#endif
                H_tmp[(row_idx*max_gmres_iters) + col_idx] = tmp;

#ifdef DEBUG_MODE_FINE
                std::cout << "I'm writing: " << tmp << " to H_tmp at index: ";
                std::cout << (row_idx*max_gmres_iters) + col_idx << std::endl;
#endif
            }
        }
    }

#ifdef DEBUG_MODE
    std::cout << "H_tmp" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << H_tmp[(max_gmres_iters*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Form Givens rotation matrix for next iteration
    // NOTE: not forming the entire matrix, just local matrix
    // NOTE: since J is typically column accessed, need to transpose to access rows
    // double J_denom = std::sqrt(std::pow(H_tmp[(iter_count*max_gmres_iters) + iter_count],2) + \
    //                  std::pow(H_tmp[(iter_count+1)*max_gmres_iters + iter_count],2));
    double J_denom = std::sqrt(std::pow(H_tmp[(iter_count*max_gmres_iters) + iter_count],2) + \
                     std::pow(H_tmp[(iter_count+1)*max_gmres_iters + iter_count],2));

    // non-transposed (not as performant(?), but easier to understand)
    // double c_i = H_tmp[(iter_count*max_gmres_iters) + iter_count] / J_denom;
    // double s_i = H_tmp[((iter_count+1)*max_gmres_iters) + iter_count] / J_denom;
    double c_i = H_tmp[(iter_count*max_gmres_iters) + iter_count] / J_denom;
    double s_i = H_tmp[((iter_count+1)*max_gmres_iters) + iter_count] / J_denom;

    // std::cout << "c_i = " << c_i << std::endl;
    // std::cout << "s_i = " << s_i << std::endl; 

    // J[0][0] locally
    J[iter_count*(max_gmres_iters+1) + iter_count] = c_i;
    // J[0][1] locally
    J[iter_count*(max_gmres_iters+1) + (iter_count+1)] = s_i;
    // J[1][0] locally
    J[(iter_count+1)*(max_gmres_iters+1) + iter_count] = -1.0 * s_i;
    // J[1][1] locally
    J[(iter_count+1)*(max_gmres_iters+1) + (iter_count+1)] = c_i;

#ifdef DEBUG_MODE
    std::cout << "J" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << J[((max_gmres_iters+1)*row_idx) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // transposed (use for row-wise accesses)
    // // c_i
    // J[0*2 + 0] = H_tmp[(iter_count*max_gmres_iters) + iter_count] / J_denom;
    // // s_i
    // J[1*2 + 0] = H_tmp[(iter_count*max_gmres_iters) + iter_count + 1] / J_denom;
    // // -s_i
    // J[0*2 + 1] = -1.0*J[1*2 + 0];
    // // c_i
    // J[1*2 + 1] = J[0*2 + 0];

// #ifdef DEBUG_MODE
//     std::cout << "J" << " = [\n";
//         for(int row_idx = 0; row_idx < 2; ++row_idx){
//             for(int col_idx = 0; col_idx < 2; ++col_idx){
//                 std::cout << J[(2)*row_idx + col_idx]  << ", ";
//             }
//             std::cout << "\n";
//         }
//     std::cout << "]" << std::endl;
// #endif

#ifdef DEBUG_MODE
    std::cout << "old_Q" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q[(max_gmres_iters+1)*row_idx + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // Combine local Givens rotations with all previous, 
    // i.e. compute Q <- J*Q (dense MMM)
    for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
            tmp = 0.0;
            // Recall, J is transposed so we can access rows here
            // We need to read and write to different Q structs to avoid conflicts
            // strided_1_dot(&Q_copy[(max_gmres_iters+1)*(row_idx + iter_count) + iter_count], &J[col_idx], &tmp, 2, max_gmres_iters);
            // strided_2_dot(&J[2*row_idx], &Q_copy[col_idx + (max_gmres_iters+1)*iter_count + iter_count], &tmp, 2, max_gmres_iters+1);
            // Q[(max_gmres_iters+1)*(row_idx + iter_count) + col_idx + iter_count] = tmp;
            // Q[(max_gmres_iters+1)*(row_idx + iter_count) + col_idx + iter_count] = tmp;
            strided_2_dot(&J[row_idx*(max_gmres_iters+1)], &Q[col_idx], &tmp, max_gmres_iters+1, max_gmres_iters+1);
            // for (int i = 0; i < (max_gmres_iters+1); ++i){
            //     tmp += Q[row_idx*(max_gmres_iters+1) + i] * J[col_idx + i*(max_gmres_iters+1)];
            //     std::cout << Q[row_idx*(max_gmres_iters+1)] << " * " << J[col_idx + i*(max_gmres_iters+1)] << std::endl;
            // }
            Q_copy[row_idx*(max_gmres_iters+1) + col_idx] = tmp;


// #ifdef DEBUG_MODE_FINE
            // std::cout << "I'm writing: " << tmp << " to Q at index: ";
            // std::cout << row_idx*(max_gmres_iters+1) + col_idx << std::endl;
// #endif
        }
    }
    // std::swap(Q_copy, Q);
    // Q = Q_copy;
    // Very lazy way!
    for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
            Q[(max_gmres_iters+1)*row_idx + col_idx] = Q_copy[(max_gmres_iters+1)*row_idx + col_idx];
        }
    }

#ifdef DEBUG_MODE
    std::cout << "new_Q" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q[(max_gmres_iters+1)*row_idx + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

    // reset R to identity matrix
    // for(int i = 0; i <= max_gmres_iters; ++i){
    //     for (int j = 0; j < max_gmres_iters; ++j){
    //         R[i*(max_gmres_iters) + j] = 0.0;
    //         if(i == j){
    //             R[i*(max_gmres_iters) + j] = 1.0;
    //         }
    //     }
    // }

    // R <- Q*H (dense MMM) (m+1 x m) <- (m+1 x m+1)(m+1 x m)
    for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
            tmp = 0.0;
            strided_2_dot(&Q[row_idx*(max_gmres_iters+1)], &H[col_idx], &tmp, max_gmres_iters+1, max_gmres_iters);

            // for (int i = 0; i < (max_gmres_iters+1); ++i){
            //     tmp += Q[row_idx*(max_gmres_iters+1) + i] * H[col_idx + i*max_gmres_iters];
            //     std::cout << Q[row_idx*(max_gmres_iters+1)+i] << " * " << H[col_idx + i*(max_gmres_iters)] << std::endl;
            // }

#ifdef DEBUG_MODE
            // Verify subdiagonal is eliminated
            // Toggle for debugging
            // for(int tmp_row_idx = 0; tmp_row_idx <= max_gmres_iters; ++tmp_row_idx){
            //     for(int tmp_col_idx = 0; tmp_col_idx < max_gmres_iters; ++tmp_col_idx){
            //         if(col_idx < row_idx){
            //             if(std::abs(tmp) > 0.0){
            //                 printf("GMRES WARNING: At index (%i, %i), R has a value %.17g, \n" \
            //                     "where is was meant to be eliminated! Forcing to 0.0.\n", row_idx, col_idx, tmp);
            //                 tmp = 0.0;
            //             }
            //         }
            //     }
            // }
#endif
            R[(row_idx*max_gmres_iters) + col_idx] = tmp;

#ifdef DEBUG_MODE_FINE
            std::cout << "I'm writing: " << tmp << " to R at index: ";
            std::cout << (row_idx*max_gmres_iters) + col_idx << std::endl;
#endif
        }
    }

#ifdef DEBUG_MODE
    std::cout << "R" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << R[(row_idx*max_gmres_iters) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;
#endif

#ifdef DEBUG_MODE
    // Sanity check: Validate that H == Q_tR ((m+1 x m) == (m+1 x m+1)(m+1 x m))

    double *Q_t = new double[(max_gmres_iters+1) * (max_gmres_iters+1)];
    init(Q_t, 0.0, (max_gmres_iters+1) * (max_gmres_iters+1));

    dense_transpose(Q, Q_t, max_gmres_iters+1, max_gmres_iters+1);

    std::cout << "Q_t" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx <= max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q_t[(max_gmres_iters+1)*row_idx + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    double *Q_tR = new double[(max_gmres_iters+1) * (max_gmres_iters)];
    init(Q_tR, 0.0, (max_gmres_iters+1) * (max_gmres_iters));

    // Compute Q_tR
    for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
            double tmp = 0.0;
            // dot(&Q[row_idx*(max_gmres_iters+1)], &R[col_idx], &tmp, max_gmres_iters);
            strided_2_dot(&Q_t[row_idx*(max_gmres_iters+1)], &R[col_idx], &tmp, max_gmres_iters+1, max_gmres_iters);
            // std::cout << "I'm writing to Q_tR at index: " << (col_idx*max_gmres_iters) + row_idx << std::endl;
            // for (int i = 0; i < max_gmres_iters; ++i){
                // tmp += Q[row_idx*(max_gmres_iters+1) + i] * R[col_idx*max_gmres_iters + i];
                // std::cout << Q[row_idx*(max_gmres_iters+1) + i] << " * " << R[col_idx*max_gmres_iters + i] << std::endl;
            // }

            Q_tR[(row_idx*max_gmres_iters) + col_idx] = tmp;
#ifdef DEBUG_MODE_FINE
            std::cout << "I'm writing: " << tmp << " to Q_tR at index: ";
            std::cout << (row_idx*max_gmres_iters) + col_idx << std::endl;
#endif
        }
    }

    // Print Q_tR
    std::cout << "Q_tR" << " = [\n";
        for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
            for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
                std::cout << std::setw(fixed_width);
                std::cout << Q_tR[(row_idx*max_gmres_iters) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    std::cout << "]" << std::endl;

    // Scan and validate H=Q_tR
    for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
        for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
            int idx = row_idx*max_gmres_iters + col_idx;
            if(std::abs(Q_tR[idx] - H[idx]) > tol){
                printf("GMRES ERROR: The Q_tR factorization of H at index %i has a value %.17g, \n \
                    and does not have a value of %.17g as was expected.\n", \
                    row_idx*max_gmres_iters + col_idx, Q_tR[idx], H[row_idx*max_gmres_iters + col_idx]);
            }
        }
    }
#endif
    // // Scale Givens rotations with beta (the initial residual norm)
    // double *g_k = new double[iter_count+1];

    // // Copy g vector information from previous iteration
    // for(int i = 0; i < iter_count; ++i){
    //     g_k[i] = g[i];
    // }

#ifdef DEBUG_MODE
    std::cout << "g_" << iter_count << " = [\n";
    for(int i = 0; i <= max_gmres_iters; ++i){
        std::cout << g[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // g_k+1 <- Q* g_k (dMVM) ((m+1 x 1) = (m+1 x m+1)(m+1 x 1))
    init(g_copy, 0.0, max_gmres_iters+1);
    g_copy[0] = beta;
    init(g, 0.0, max_gmres_iters+1);
    g[0] = beta;
    for(int row_idx = 0; row_idx < max_gmres_iters+1; ++row_idx){
        tmp = 0.0;
        dot(&Q[row_idx*(max_gmres_iters+1)], g, &tmp, max_gmres_iters+1);
        // for (int i = 0; i < max_gmres_iters+1; ++i){
        //     tmp += Q[row_idx*(max_gmres_iters+1) + i] * g[i];
        //     std::cout << Q[row_idx*(max_gmres_iters+1) + i] << " * " << g[i] << std::endl;
        // }
        g_copy[row_idx] = tmp;
    }

    // Very lazy !
    for(int row_idx = 0; row_idx < max_gmres_iters+1; ++row_idx){
        g[row_idx] = g_copy[row_idx];
    }

#ifdef DEBUG_MODE
    std::cout << "g_" << iter_count+1 << " = [\n";
    for(int i = 0; i <= max_gmres_iters; ++i){
        std::cout << g[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // Extract the last element from g as residual norm
    *residual_norm = std::abs(g[iter_count + 1]);

#ifdef DEBUG_MODE
    std::cout << "residual_norm = " << *residual_norm << std::endl;
#endif

    // TODO: just allocate in preprocessing
    delete J;
    delete H_tmp;
    // delete Q_copy;

#ifdef DEBUG_MODE
    delete Q_tR;
#endif
    // delete g_k;

    // if (iter_count == 1){
    //    exit(1); 
    // }

    // exit(1);
}

void solve_cpu(
    argType *args
){
    int n_rows = args->sparse_mat->crs_mat->n_rows;
    // Unpack relevant args
    // double *x = args->x_old; // GS and GMRES
    // double *x_new = args->x_new; // Jacobi
    // double *x_old = args->x_old; // Jacobi and (sometimes) GMRES
    double *x = new double[n_rows];
    double *x_new = new double[n_rows];
    double *x_old = new double[n_rows];

    // Need deep copy
    for(int i = 0; i < n_rows; ++i){
        x[i] = args->x_old[i];
        x_new[i] = args->x_new[i];
        x_old[i] = args->x_old[i];
    }

    // Not sure about these
    double *tmp = args->tmp;
    double *D = args->D;
    double *r = args->r;
    double *b = args->b;
    SparseMtxFormat *sparse_mat = args->sparse_mat;
    Flags *flags = args->flags;

    double residual_norm;

    if(args->flags->print_iters){
        iter_output(x, args->vec_size, args->loop_params->iter_count);
        printf("\n");
    }

    // TODO: should put in preprocessing
    // NOTE: only temporary for mem allocation, for testing
    // Restart length can take the place of this I guess
    int max_gmres_iters = 100;
    // ^ For testing purposes, we only search the first 10 krylov dims
    // TODO: only used for GMRES... so should guard somehow
    double *V = new double[sparse_mat->crs_mat->n_rows * max_gmres_iters]; // (m x n)
    init(V, 0.0, sparse_mat->crs_mat->n_rows * max_gmres_iters);
    // Give v0 to first row of V
    for(int i = 0; i < sparse_mat->crs_mat->n_rows; ++i){
        V[i] = args->init_v[i];
    }
    
    double *H = new double[(max_gmres_iters+1) * max_gmres_iters]; // (m+1 x m) 
    init(H, 0.0, max_gmres_iters * (max_gmres_iters+1));
    //  (^transposed to store as row vectors instead)

    double *R_outer = new double[max_gmres_iters * (max_gmres_iters+1)];
    init(R_outer, 0.0, max_gmres_iters * (max_gmres_iters+1));
    // Initialize 1s in identity matrix
    for(int i = 0; i <= max_gmres_iters; ++i){
        for (int j = 0; j < max_gmres_iters; ++j){
            if(i == j){
                R_outer[i*(max_gmres_iters+1) + j] = 1.0;
            }
        }
    }

    double *Q = new double[(max_gmres_iters+1) * (max_gmres_iters+1)]; // (m+1 x m+1)
    init(Q, 0.0, (max_gmres_iters+1) * (max_gmres_iters+1));

    double *Q_copy = new double[(max_gmres_iters+1) * (max_gmres_iters+1)]; // (m+1 x m+1)
    init(Q_copy, 0.0, (max_gmres_iters+1) * (max_gmres_iters+1));

    // Initialize 1s in identity matrix
    for(int i = 0; i <= max_gmres_iters; ++i){
        for (int j = 0; j <= max_gmres_iters; ++j){
            if(i == j){
                Q[i*(max_gmres_iters+1) + j] = 1.0;
                Q_copy[i*(max_gmres_iters+1) + j] = 1.0;
            }
        }
    }

    double *g = new double[max_gmres_iters+1];
    init(g, 0.0, max_gmres_iters+1);
    g[0] = args->beta; // <- supply starting element
    double *g_copy = new double[max_gmres_iters+1];
    init(g_copy, 0.0, max_gmres_iters+1);
    g_copy[0] = args->beta; // <- supply starting element
    

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << x[i] << std::endl;
    }
#endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        // TODO: really lazy, and is only "needed" for GMRES. Put somewhere else!!
        // Only the upper triangular part of H (again stored as row vectors, so transposed)
        // double *R = new double[(args->loop_params->iter_count-1) * args->loop_params->iter_count];

        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            jacobi_iteration_ref_cpu(sparse_mat, D, b, x_old, x_new);
            // jacobi_iteration_sep_cpu(sparse_mat, D, b, x_old, x_new, args->vec_size);
        }
        else if(args->solver_type == "gauss-seidel"){
            // For a reference solution, not meant for use with USpMV library
            gs_iteration_ref_cpu(sparse_mat, tmp, D, b, x);
            // gs_iteration_sep_cpu(sparse_mat, tmp, D, b, x, args->vec_size);
        }
        else if(args->solver_type == "gmres"){
            gm_iteration_ref_cpu(
                sparse_mat, 
                V,
                H,
                Q,
                Q_copy,
                tmp, 
                R_outer,
                g,
                g_copy,
                b, 
                x,
                args->beta,
                args->vec_size,
                args->loop_params->iter_count,
                &residual_norm,
                max_gmres_iters
            );
        }
        
        // GMRES does not compute x each iteration by default
        if(args->solver_type == "jacobi" || args->solver_type == "gauss-seidel"){
            if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
                
                // Record residual every "residual_check_len" iterations
                if(args->solver_type == "jacobi"){
                    calc_residual_cpu(sparse_mat, x_new, b, r, tmp, args->vec_size);
                }
                else if(args->solver_type == "gauss-seidel"){
                    calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
                }
                
                residual_norm = infty_vec_norm_cpu(r, args->vec_size);
                args->normed_residuals[args->loop_params->residual_count] = residual_norm;
                ++args->loop_params->residual_count;

                if(flags->print_iters){
                    if(args->solver_type == "jacobi"){
                        iter_output(x_new, args->vec_size, args->loop_params->iter_count);
                    }
                    else if(args->solver_type == "gauss-seidel"){
                        iter_output(x, args->vec_size, args->loop_params->iter_count);
                    }
                }
            }
        }

#ifdef DEBUG_MODE
// TODO: seems like redundant functionality with what is directly above...
        std::cout << "x_" << args->loop_params->iter_count+1 <<" = [";
        if(args->solver_type == "jacobi"){
            for(int i = 0; i < args->vec_size; ++i){
                std::cout << x_new[i] << ", ";
            }
        }
        else if (args->solver_type == "gauss-seidel"){
            for(int i = 0; i < args->vec_size; ++i){
                std::cout << x[i] << ", ";
            }
        }
        else if(args->solver_type == "gmres"){
            // // Take all but the last row of V
            gmres_get_x(R_outer, g, x, x_old, V, sparse_mat->crs_mat->n_rows, args->loop_params->iter_count, max_gmres_iters);
            // for(int i = 0; i < args->vec_size; ++i){
            //     std::cout << x[i] << ", ";
            // }
        }
        std::cout << "]" << std::endl;
  
        if(args->solver_type == "gmres"){
            calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
            std::cout << "computed residual norm: " << euclidean_vec_norm_cpu(r, args->vec_size) << std::endl;
        }
        else{
            // Might not be computed here
            std::cout << "computed residual norm: " << infty_vec_norm_cpu(r, args->vec_size) << std::endl;
        }
        std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
#endif  

        if(args->solver_type == "jacobi"){
            std::swap(x_new, x_old);
        }
    
        ++args->loop_params->iter_count;
        // if(residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters){
        //     // A very lazy way of saving the "upper triangular" matrix 
        //     std::swap(R_outer, R);
        // }

        // delete R;

    } while(
        args->loop_params->iter_count < n_rows &&
        residual_norm > args->loop_params->stopping_criteria && 
        args->loop_params->iter_count < args->loop_params->max_iters
    );

    args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    if(args->solver_type == "jacobi"){
        std::swap(x_old, args->x_star);
    }
    else if (args->solver_type == "gauss-seidel"){
        std::swap(x, args->x_star);
    }    
    
    // Record final residual with approximated solution vector x
    if(args->solver_type == "gmres"){
        // TODO: store upper triangular of H in a better place
        // By default, GMRES doesn't compute actual x approximation until after convergence
        gmres_get_x(R_outer, g, args->x_star, x_old, V, sparse_mat->crs_mat->n_rows, args->loop_params->iter_count - 1, max_gmres_iters);
        std::cout << "x_star = [";
        for(int i = 0; i < args->vec_size; ++i){
                std::cout << args->x_star[i] << ", ";
            }
        std::cout << "]" << std::endl;
        calc_residual_cpu(sparse_mat, args->x_star, b, r, tmp, args->vec_size);
        args->normed_residuals[args->loop_params->residual_count] = euclidean_vec_norm_cpu(r, args->vec_size);
        std::cout << "args->normed_residuals[args->loop_params->residual_count] = " << args->normed_residuals[args->loop_params->residual_count] <<std::endl;
    }
    else{
        calc_residual_cpu(sparse_mat, args->x_star, b, r, tmp, args->vec_size);

        args->normed_residuals[args->loop_params->residual_count] = infty_vec_norm_cpu(r, args->vec_size);
        std::cout << "args->normed_residuals[args->loop_params->residual_count] = " << args->normed_residuals[args->loop_params->residual_count] <<std::endl;
    }


#ifdef USE_USPMV
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[args->vec_size];
    apply_permutation(x_star_perm, args->x_star, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);

    // Deep copy, so you can free memory
    // TODO: wrap in func
    for(int i = 0; i < args->vec_size; ++i){
        args->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
#endif

    if(args->solver_type == "gmres"){
        delete V;
        delete H;
        delete Q;
        delete Q_copy;
        delete g;
        delete g_copy;
        delete R_outer;
    }

    delete x;
    delete x_new;
    delete x_old;

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);
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
#ifndef __CUDACC__
    solve_cpu(args);
#else
    solve_gpu(args);
#endif
}