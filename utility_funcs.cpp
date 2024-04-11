#include <stdio.h>
#include <iostream>
#include <random>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <math.h>
#include <sys/time.h>
#include <cstring>
#include <cmath>
#include <vector>

#ifdef USE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#include "kernels.hpp"

void generate_vector(
    std::vector<double> *vec_to_populate,
    int size,
    bool rand_flag,
    double initial_val
){
    if(rand_flag){
        double upper_bound = 50000;
        double lower_bound = 0;
        srand(time(nullptr));

        double range = (upper_bound - lower_bound); 
        double div = RAND_MAX / range;

        for(int i = 0; i < size; ++i){
            (*vec_to_populate)[i] = lower_bound + (rand() / div); //NOTE: expensive?
        }
    }
    else{
        for(int i = 0; i < size; ++i){
            (*vec_to_populate)[i] = initial_val;
        }
    }

}

double infty_vec_norm(
    const std::vector<double> *vec
){
    double max_abs = 0.;
    double curr_abs;
    for (int i = 0; i < vec->size(); ++i){
        curr_abs = std::abs((*vec)[i]);
        if ( curr_abs > max_abs){
            max_abs = curr_abs; 
        }
    }

    return max_abs;
}

double infty_mat_norm(
    const CRSMtxData *crs_mat
){
    // Accumulate with sum all the elements in each row
    std::vector<double> row_sums(crs_mat->n_rows, 0.0);

    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx)
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
#ifdef DEBUG_MODE
            std::cout << "summing: " << crs_mat->val[nz_idx] << " in infty mat norm" << std::endl;
#endif
            // row_sums[row_idx] += abs(crs_mat->val[nz_idx]); 
            row_sums[row_idx] += crs_mat->val[nz_idx]; 

        }

    // The largest sum is the matrix infty norm
    return infty_vec_norm(&row_sums);
}

/* Residual here is the distance from A*x_new to b, where the norm
 is the infinity norm: ||A*x_new-b||_infty */
void calc_residual(
    const CRSMtxData *crs_mat,
    const std::vector<double> *x,
    const std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *A_x_tmp
){
    spmv_crs(A_x_tmp, crs_mat, x);

    subtract_vectors(r, b, A_x_tmp);
}

void start_time(
    timeval *begin
){
    gettimeofday(begin, 0);
}

double end_time(
    timeval *begin,
    timeval *end
){
    gettimeofday(end, 0);
    long seconds = end->tv_sec - begin->tv_sec;
    long microseconds = end->tv_usec - begin->tv_usec;
    return seconds + microseconds*1e-6;
}

void recip_elems(
    std::vector<double> *recip_vec,
    std::vector<double> *vec
){

    // Sanity check. NOTE: How bad for performance is this?
    if (std::find(vec->begin(), vec->end(), 0) != vec->end()){
        printf("ERROR: recip_elems: Zero detected.\n");
        exit(1);
    }

    // NOTE: changes vec "in-place"
    for(int i = 0; i < vec->size(); ++i){
        (*recip_vec)[i] = 1/(*vec)[i];
    }
}

void gen_neg_inv(
    std::vector<double> *neg_inv_coo_vec,
    std::vector<double> *inv_coo_vec,
    std::vector<double> *coo_vec
){
    // Recipricate elements in vector
    recip_elems(inv_coo_vec, coo_vec);

    // Copy recipricol elements to another vector
    (*neg_inv_coo_vec) = (*inv_coo_vec); 

    // Negate this vector
    std::transform(
        neg_inv_coo_vec->cbegin(), 
        neg_inv_coo_vec->cend(), 
        neg_inv_coo_vec->begin(), 
        std::negate<double>()
    );
}

void extract_diag(
    const CRSMtxData *crs_mat,
    std::vector<double> *diag
){
    #pragma omp parallel for schedule (static)
    for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx){
        for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
            if(row_idx == crs_mat->col[nz_idx]){
                (*diag)[row_idx] = crs_mat->val[nz_idx];
                break;
            }
        }
    }
}

void compare_with_direct(
    CRSMtxData *crs_mat,
    std::string matrix_file_name,
    LoopParams loop_params,
    std::vector<double> *x_star,
    double iterative_final_residual
){
    
#ifdef USE_EIGEN
    Eigen::SparseMatrix<double> A;
    Eigen::loadMarket(A, matrix_file_name);
    A.makeCompressed();
    // Just keep crs matrix from before?

    int eigen_n_rows = static_cast<int>(A.rows());
    int eigen_n_cols = static_cast<int>(A.cols());
    int eigen_nnz = static_cast<int>(A.nonZeros());

    std::vector<double> b_vec(eigen_n_cols);

    generate_vector(&b_vec, eigen_n_cols, false, loop_params.init_b);
    // ^ b should likely draw from A(min) to A(max) range of values

    Eigen::VectorXd b = Eigen::VectorXd::Map(&b_vec[0], b_vec.size());

    // NOTE: no initial guess with a direct solver
    // Eigen::VectorXd x_direct(x_vec.size());

    // solve Ax = b
    Eigen::SparseLU<Eigen::SparseMatrix<double> > solver;

    solver.analyzePattern(A); 

    solver.factorize(A); 

    if(solver.info() != Eigen::Success) {
        printf("ERROR: eigen library decomposition failed.\n");
        exit(1);
    return;
    }

    Eigen::VectorXd x_direct = solver.solve(b);

    if(solver.info() != Eigen::Success) {
        printf("ERROR: eigen library direct solve failed.\n");
        exit(1);
    return;
    }

    std::vector<double> x_direct_vec(&x_direct[0], x_direct.data()+x_direct.cols()*x_direct.rows());

    double direct_final_residual = calc_residual(crs_mat, &x_direct_vec, &b_vec);
    
    write_comparison_to_file(x_star, iterative_final_residual, &x_direct_vec, direct_final_residual);
#else
    printf("ERROR: eigen library not correctly linked, cannot compare approximation with direct solver. Check flags.\n");
    exit(1);
#endif
}

void split_L_U(
    COOMtxData *full_coo_mtx,
    COOMtxData *L_coo_mtx,
    COOMtxData *U_coo_mtx
){
    bool explitit_zero_warning_flag = false;
    int L_coo_mtx_count = 0;
    int U_coo_mtx_count = 0;
    int D_coo_vec_count = 0;

    // Force same dimensions for consistency
    U_coo_mtx->n_rows = full_coo_mtx->n_rows;
    U_coo_mtx->n_cols = full_coo_mtx->n_cols;
    U_coo_mtx->is_sorted = full_coo_mtx->is_sorted;
    U_coo_mtx->is_symmetric = false;
    L_coo_mtx->n_rows = full_coo_mtx->n_rows;
    L_coo_mtx->n_cols = full_coo_mtx->n_cols;
    L_coo_mtx->is_sorted = full_coo_mtx->is_sorted;
    L_coo_mtx->is_symmetric = false;

    for(int nz_idx = 0; nz_idx < full_coo_mtx->nnz; ++nz_idx){
        // If column and row less than i, this nz is in the L matrix
        if(full_coo_mtx->J[nz_idx] < full_coo_mtx->I[nz_idx]){
            // Copy element to lower matrix
            L_coo_mtx->I.push_back(full_coo_mtx->I[nz_idx]);
            L_coo_mtx->J.push_back(full_coo_mtx->J[nz_idx]);
            L_coo_mtx->values.push_back(full_coo_mtx->values[nz_idx]);
            ++L_coo_mtx->nnz;
            // std::cout << full_coo_mtx->values[nz_idx] << " sent to lower matrix" << std::endl;
        }
        else if(full_coo_mtx->J[nz_idx] > full_coo_mtx->I[nz_idx]){
            // Copy element to upper matrix
            U_coo_mtx->I.push_back(full_coo_mtx->I[nz_idx]);
            U_coo_mtx->J.push_back(full_coo_mtx->J[nz_idx]);
            U_coo_mtx->values.push_back(full_coo_mtx->values[nz_idx]);
            ++U_coo_mtx->nnz;
            // std::cout << full_coo_mtx->values[nz_idx] << " sent to upper matrix" << std::endl;
        }
        else if(full_coo_mtx->I[nz_idx] == full_coo_mtx->J[nz_idx]){
        //     // Copy element to vector representing diagonal matrix
        //     // NOTE: Don't need push_back because we know the size
            if(abs(full_coo_mtx->values[nz_idx]) < 1e-15 && !explitit_zero_warning_flag){ // NOTE: error tolerance too tight?
                printf("WARNING: split_upper_lower_diagonal: explicit zero detected on diagonal at nz_idx %i.\n"
                        "row: %i, col: %i, val: %f.\n", nz_idx, full_coo_mtx->I[nz_idx], full_coo_mtx->J[nz_idx], full_coo_mtx->values[nz_idx]);
        //         explitit_zero_warning_flag = true;
            }
        //     (*D_coo_vec)[D_coo_vec_count] = full_coo_mtx->values[nz_idx];
            ++D_coo_vec_count;
        }
        else{
            printf("ERROR: split_upper_lower_diagonal: nz_idx %i cannot be segmented.\n", nz_idx);
            exit(1);
        }
    }

    // Sanity checks; TODO: Make optional
    // All elements from full_coo_mtx need to be accounted for
    int copied_elems_count = L_coo_mtx->nnz + U_coo_mtx->nnz + D_coo_vec_count; 
    if(copied_elems_count != full_coo_mtx->nnz){
        printf("ERROR: split_upper_lower_diagonal: only %i out of %i elements were copied from full_coo_mtx.\n", copied_elems_count, full_coo_mtx->nnz);
        exit(1);
    }

}

// TODO: MPI preprocessing will go here
void preprocessing(
    COOMtxData *coo_mat,
    CRSMtxData *crs_mat,
    CRSMtxData *crs_L,
    CRSMtxData *crs_U,
    std::vector<double> *x_star,
    std::vector<double> *x_new,
    std::vector<double> *x_old,
    std::vector<double> *A_x_tmp,
    std::vector<double> *D,
    std::vector<double> *r,
    std::vector<double> *b,
    LoopParams *loop_params,
    const std::string solver_type
){
    if(solver_type == "gauss-seidel"){
        COOMtxData *coo_L = new COOMtxData;
        COOMtxData *coo_U = new COOMtxData;

        split_L_U(coo_mat, coo_L, coo_U);

        coo_L->convert_to_crs(crs_L);
        coo_U->convert_to_crs(crs_U);

        delete coo_L;
        delete coo_U;
    }

    coo_mat->convert_to_crs(crs_mat);

    extract_diag(crs_mat, D);

    x_star->resize(crs_mat->n_cols, 0);
    b->resize(crs_mat->n_cols, 0);

    // TODO: What are the ramifications of having x and b different scales than the data? And how to make the "same scale" as data?
    // Make b vector
    generate_vector(b, crs_mat->n_cols, false, loop_params->init_b);
    // ^ b should likely draw from A(min) to A(max) range of values

    // Make initial x vector
    generate_vector(x_old, crs_mat->n_cols, false, loop_params->init_x);

    // Precalculate stopping criteria
    calc_residual(crs_mat, x_old, b, r, A_x_tmp);
    loop_params->stopping_criteria = loop_params->tol * infty_vec_norm(r); 
}