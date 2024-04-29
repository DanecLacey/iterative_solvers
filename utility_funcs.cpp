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
    std::vector<double> *values,
    double initial_val
){
    if(rand_flag){
        double upper_bound = *(std::max_element(std::begin(*values), std::end(*values)));
        double lower_bound = *(std::min_element(std::begin(*values), std::end(*values)));
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

// TODO: use case for making this device resident? Or using it at all?
// double infty_mat_norm(
//     const CRSMtxData *crs_mat
// ){
//     // Accumulate with sum all the elements in each row
//     std::vector<double> row_sums(crs_mat->n_rows, 0.0);

//     for(int row_idx = 0; row_idx < crs_mat->n_rows; ++row_idx)
//         for(int nz_idx = crs_mat->row_ptr[row_idx]; nz_idx < crs_mat->row_ptr[row_idx+1]; ++nz_idx){
// #ifdef DEBUG_MODE
//             std::cout << "summing: " << crs_mat->val[nz_idx] << " in infty mat norm" << std::endl;
// #endif
//             // row_sums[row_idx] += abs(crs_mat->val[nz_idx]); 
//             row_sums[row_idx] += crs_mat->val[nz_idx]; 

//         }

//     // The largest sum is the matrix infty norm
//     return infty_vec_norm_cpu(&row_sums);
// }


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
    const COOMtxData *coo_mat,
    std::vector<double> *diag
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
            (*diag)[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
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
    printf("ERROR: eigen library depreciated (for now).\n");
    exit(1);
    Eigen::SparseMatrix<double> A;
    Eigen::loadMarket(A, matrix_file_name);
    A.makeCompressed();
    // Just keep crs matrix from before?

    int eigen_n_rows = static_cast<int>(A.rows());
    int eigen_n_cols = static_cast<int>(A.cols());
    int eigen_nnz = static_cast<int>(A.nonZeros());

    std::vector<double> b_vec(eigen_n_cols);

    // TODO: fix signature
    // generate_vector(&b_vec, eigen_n_cols, false, loop_params.init_b);
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


void convert_to_crs(
    COOMtxData *coo_mat,
    CRSMtxData *crs_mat
    )
{
    crs_mat->n_rows = coo_mat->n_rows;
    crs_mat->n_cols = coo_mat->n_cols;
    crs_mat->nnz = coo_mat->nnz;

    crs_mat->row_ptr = new int[crs_mat->n_rows+1];
    int *nnzPerRow = new int[crs_mat->n_rows];

    crs_mat->col = new int[crs_mat->nnz];
    crs_mat->val = new double[crs_mat->nnz];

    for(int idx = 0; idx < crs_mat->nnz; ++idx)
    {
        crs_mat->col[idx] = coo_mat->J[idx];
        crs_mat->val[idx] = coo_mat->values[idx];
    }

    for(int i = 0; i < crs_mat->n_rows; ++i)
    { 
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(int i=0; i < crs_mat->nnz; ++i)
    {
        ++nnzPerRow[coo_mat->I[i]];
    }

    crs_mat->row_ptr[0] = 0;
    for(int i=0; i < crs_mat->n_rows; ++i)
    {
        crs_mat->row_ptr[i+1] = crs_mat->row_ptr[i]+nnzPerRow[i];
    }

    if(crs_mat->row_ptr[crs_mat->n_rows] != crs_mat->nnz)
    {
        printf("ERROR: converting to CRS.\n");
        exit(1);
    }

    delete[] nnzPerRow;
}

// TODO: MPI preprocessing will go here
void preprocessing(
    argType *args
){
    if(args->solver_type == "gauss-seidel"){
        COOMtxData *coo_L = new COOMtxData;
        COOMtxData *coo_U = new COOMtxData;

        split_L_U(args->coo_mat, coo_L, coo_U);

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
        convert_to_scs<double, int>(mtx_L, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_L);

        MtxData<double, int> *mtx_U = new MtxData<double, int>;
        mtx_U->n_rows = coo_U->n_rows;
        mtx_U->n_cols = coo_U->n_cols;
        mtx_U->nnz = coo_U->nnz;
        mtx_U->is_sorted = true; //TODO
        mtx_U->is_symmetric = false; //TODO
        mtx_U->I = coo_U->I;
        mtx_U->J = coo_U->J;
        mtx_U->values = coo_U->values;
        convert_to_scs<double, int>(mtx_U, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_U);
#endif
        convert_to_crs(coo_L, args->sparse_mat->crs_L);
        convert_to_crs(coo_U, args->sparse_mat->crs_U);

        delete coo_L;
        delete coo_U;
    }

#ifdef USE_USPMV
    MtxData<double, int> *mtx_mat = new MtxData<double, int>;
    mtx_mat->n_rows = args->coo_mat->n_rows;
    mtx_mat->n_cols = args->coo_mat->n_cols;
    mtx_mat->nnz = args->coo_mat->nnz;
    mtx_mat->is_sorted = true; //TODO
    mtx_mat->is_symmetric = false; //TODO
    mtx_mat->I = args->coo_mat->I;
    mtx_mat->J = args->coo_mat->J;
    mtx_mat->values = args->coo_mat->values;

    // NOTE: Symmetric permutation, i.e. rows and columns
    convert_to_scs(mtx_mat, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat);
    permute_scs_cols(args->sparse_mat->scs_mat, &(args->sparse_mat->scs_mat->old_to_new_idx)[0]);
    args->vec_size = args->sparse_mat->scs_mat->n_rows_padded;
    // args->sparse_mat->scs_mat->print();
    // exit(1);
#else
    args->vec_size = args->coo_mat->n_cols;
#endif

    convert_to_crs(args->coo_mat, args->sparse_mat->crs_mat);
    
#ifdef __CUDACC__
    cudaMalloc(&(args->d_row_ptr), (args->sparse_mat->n_rows+1)*sizeof(int));
    cudaMalloc(&(args->d_col), (args->sparse_mat->nnz)*sizeof(int));
    cudaMalloc(&(args->d_val), (args->sparse_mat->nnz)*sizeof(double));
    cudaMemcpy(args->d_row_ptr, &(args->sparse_mat->row_ptr)[0], (args->sparse_mat->n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_col, &(args->sparse_mat->col)[0], (args->sparse_mat->nnz)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_val, &(args->sparse_mat->val)[0], (args->sparse_mat->nnz)*sizeof(double), cudaMemcpyHostToDevice);
#endif

    // Resize all working arrays, now we know the right size
    args->x_star->resize(args->vec_size, 0.0);
    args->x_new->resize(args->vec_size, 0.0);
    args->x_old->resize(args->vec_size, 0.0);
    args->tmp->resize(args->vec_size, 0.0);
    args->D->resize(args->vec_size, 0.0);
    args->r->resize(args->vec_size, 0.0);
    args->b->resize(args->vec_size, 0.0);

#ifdef __CUDACC__
    // TODO: Is this legal to do here, and free later?
    cudaMalloc(&(args->d_x_star), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_new), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_old), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_tmp), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_D), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_r), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_b), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_normed_residuals), (loop_params.max_iters / loop_params.residual_check_len + 1)*sizeof(double));
#endif

    extract_diag(args->coo_mat, args->D);

    // TODO: What are the ramifications of having x and b different scales than the data? And how to make the "same scale" as data?
    // Make b vector
    generate_vector(args->b, args->vec_size, args->flags->random_data, &(args->coo_mat->values), args->loop_params->init_b);
    // ^ b should likely draw from A(min) to A(max) range of values

    // Make initial x vector
    generate_vector(args->x_old, args->vec_size, args->flags->random_data, &(args->coo_mat->values), args->loop_params->init_x);

#ifdef USE_USPMV
    // Need to permute these vectors in accordance with SIGMA if using USpMV library
    std::vector<double> D_perm(args->vec_size, 0);
    apply_permutation(&(D_perm)[0], &(*args->D)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
    std::swap(D_perm, (*args->D));

    std::vector<double> b_perm(args->vec_size, 0);
    apply_permutation(&(b_perm)[0], &(*args->b)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
    std::swap(b_perm, (*args->b));

    // NOTE: Permuted w.r.t. columns due to symmetric permutation
    std::vector<double> x_old_perm(args->vec_size, 0);
    apply_permutation(&(x_old_perm)[0], &(*args->x_old)[0], &(args->sparse_mat->scs_mat->new_to_old_idx)[0], args->vec_size);
    std::swap(x_old_perm, *(args->x_old));
#endif

#ifdef __CUDACC__
    // NOTE: Really only need to copy x_old, D, and b data, as all other host vectors are just zero at this point?
    cudaMemcpy(args->d_x_star, &(args->x_star)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_x_new, &(args->x_new)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_x_old, &(args->x_old)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_tmp, &(args->tmp)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_D, &(args->D)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_r, &(args->r)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_b, &(args->b)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
#endif

    // Precalculate stopping criteria
    // Easier to just do on the host for now
    calc_residual_cpu(args->sparse_mat, args->x_old, args->b, args->r, args->tmp);

#ifdef DEBUG_MODE
    printf("initial residual = [");
    for(int i = 0; i < args->r->size(); ++i){
        std::cout << (*args->r)[i] << ",";
    }
    printf("]\n");
#endif

#ifndef __CUDACC__
    args->loop_params->stopping_criteria = args->loop_params->tol * infty_vec_norm_cpu(args->r); 
#else
    // The first residual is computed on the host, and given to the device
    // Easier to just do on the host for now, and give stopping criteria to device
    args->loop_params->stopping_criteria = args->loop_params->tol * infty_vec_norm_cpu(args->r); 
    cudaMalloc(&(args->loop_params->d_stopping_criteria), sizeof(double));
    cudaMemcpy(args->loop_params->d_stopping_criteria, &(args->loop_params->stopping_criteria)[0], args->vec_size*sizeof(double), cudaMemcpyHostToDevice);

#endif
}