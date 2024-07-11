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
#include <iomanip>
#include <set>

#ifdef USE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#ifdef USE_SCAMAC
#include "scamac.h"
#endif

#include "kernels.hpp"
#include "mmio.h"

void init(
    double *vec,
    double val,
    int size
){
    // TODO: validate first touch policy?
    #pragma omp parallel for
    for(int i = 0; i < size; ++i){
        vec[i] = val;
    }
}

void generate_vector(
    double *vec_to_populate,
    int size,
    bool rand_flag,
    double *values,
    double initial_val
){
    if(rand_flag){
        // TODO: Make proportional to matrix data
        // double upper_bound = *(std::max_element(std::begin(*values), std::end(*values)));
        // double lower_bound = *(std::min_element(std::begin(*values), std::end(*values)));
        double upper_bound = 10;
        double lower_bound = -10;
        srand(time(nullptr));

        double range = (upper_bound - lower_bound); 
        double div = RAND_MAX / range;

        for(int i = 0; i < size; ++i){
            vec_to_populate[i] = lower_bound + (rand() / div); //NOTE: expensive?
        }
    }
    else{
        for(int i = 0; i < size; ++i){
            vec_to_populate[i] = initial_val;
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
    double *diag
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
            diag[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
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

void gmres_get_x(
    double *R,
    double *g,
    double *x,
    double *x_0,
    double *V,
    int n_rows,
    int restart_count,
    int iter_count,
    int max_gmres_iters
){
    // double *y = new double[max_gmres_iters];
    // init(y, 0.0, max_gmres_iters);
    // double *Vy = new double[n_rows];
    // init(Vy, 0.0, n_rows);
    std::vector<double> y(max_gmres_iters, 0.0);
    std::vector<double> Vy(n_rows, 0.0);

    double diag_elem = 0.0;
    double sum;

    // Adjust for restarting
    iter_count -= restart_count*max_gmres_iters;

#ifdef DEBUG_MODE
    // Lazy way to account for restarts
    std::cout << "gmres_get_x iter_count = " << iter_count << " -> ";
    // ^ To get around weird "iter-1" problem for now
    std::cout << "gmres_get_x iter_count = " << iter_count << std::endl;
    std::cout << "gmres_get_x restart_count = " << restart_count << std::endl;
#endif

    // Could probably optimize col_idxs/2
    // Can only solve first "iter_count+1 rows"
    // for(int row_idx = 0; row_idx <= iter_count; ++row_idx){
    //     sum = 0.0;
    //     for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
    //         if(row_idx == col_idx){
    //             diag_elem = R[(row_idx*max_gmres_iters) + col_idx];
    //         }
    //         sum += R[(row_idx*max_gmres_iters) + col_idx] * y[row_idx];
    //     }
    //     y[row_idx] = (g[row_idx] - sum) / diag_elem;
    // }

#ifdef DEBUG_MODE
    std::cout << "when solving for x, R" << " = [\n";
    for(int row_idx = iter_count; row_idx >= 0; --row_idx){
        for(int col_idx = iter_count; col_idx >= 0; --col_idx){
                std::cout << std::setw(11);
                std::cout << R[(row_idx*max_gmres_iters) + col_idx]  << ", ";
            }
            std::cout << "\n";
        }
    // for(int row_idx = 0; row_idx <= max_gmres_iters; ++row_idx){
    //     for(int col_idx = 0; col_idx < max_gmres_iters; ++col_idx){
    //         std::cout << std::setw(11);
    //         std::cout << R[(row_idx*max_gmres_iters) + col_idx]  << ", ";
    //     }
    //     std::cout << "\n";
    // }

    std::cout << "]" << std::endl;
#endif

    // (dense) Backward triangular solve Ry = g ((m+1 x m)(m x 1) = (m+1 x 1))
    // Traverse R \in \mathbb{R}^(m+1 x m) from last to first row
    for(int row_idx = iter_count; row_idx >= 0; --row_idx){
        sum = 0.0;
        for(int col_idx = row_idx; col_idx < max_gmres_iters; ++col_idx){
            if(row_idx == col_idx){
                diag_elem = R[(row_idx*max_gmres_iters) + col_idx];
            }
            else{
                sum += R[(row_idx*max_gmres_iters) + col_idx] * y[col_idx];
            }
            
        }
        y[row_idx] = (g[row_idx] - sum) / diag_elem;
#ifdef DEBUG_MODE_FINE
        std::cout << g[row_idx] << " - " << sum << " / " << diag_elem << std::endl; 
#endif
    }

#ifdef DEBUG_MODE
    std::cout << "y_" << iter_count << " = [\n";
    for(int i = 0; i < max_gmres_iters; ++i){
        std::cout << y[i]  << ", ";
    }
    std::cout << "]" << std::endl;
#endif

    // (dense) matrix vector multiply Vy <- V*y ((n x 1) = (n x m)(m x 1))
    for(int col_idx = 0; col_idx < n_rows; ++col_idx){
        double tmp = 0.0;
        strided_1_dot(&V[col_idx], &y[0], &tmp, max_gmres_iters, n_rows);
        // for (int i = 0; i < max_gmres_iters; ++i){
        //     tmp += V[i*n_rows + col_idx] * y[i];
        //     // std::cout << V[i*n_rows + col_idx] << " * " << y[i] << std::endl; 
        // }
    Vy[col_idx] = tmp;
    }

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

    // delete y;
    // delete Vy;
}

void init_gmres_structs(
    argType *args,
    int n_rows
){
    int restart_len = args->gmres_restart_len;
        
    init(args->V, 0.0, n_rows * (restart_len+1));

    // Give v0 to first row of V
    for(int i = 0; i < n_rows; ++i){
        args->V[i] = args->init_v[i];
    }
    
    init(args->H, 0.0, restart_len * (restart_len+1));
    
    init(args->R, 0.0, restart_len * (restart_len+1));

    // TODO: wrap up in something cleaner
    // Initialize 1s in identity matrix
    for(int i = 0; i <= restart_len; ++i){
        for (int j = 0; j < restart_len; ++j){
            if(i == j){
                args->R[i*(restart_len+1) + j] = 1.0;
            }
        }
    }
    
    init(args->Q, 0.0, (restart_len+1) * (restart_len+1));
    init(args->Q_copy, 0.0, (restart_len+1) * (restart_len+1));

    // TODO: wrap up in something cleaner
    // Initialize 1s in identity matrix
    for(int i = 0; i <= restart_len; ++i){
        for (int j = 0; j <= restart_len; ++j){
            if(i == j){
                args->Q[i*(restart_len+1) + j] = 1.0;
                args->Q_copy[i*(restart_len+1) + j] = 1.0;
            }
        }
    }

    init(args->g, 0.0, restart_len+1);
    args->g[0] = args->beta; // <- supply starting element
    
    init(args->g_copy, 0.0, restart_len+1);
    args->g_copy[0] = args->beta; // <- supply starting element
}

void record_residual_norm(
    argType *args,
    Flags *flags,
    SparseMtxFormat *sparse_mat,
    double *residual_norm,
    double *r,
    double *x,
    double *b,
    double *x_new,
    double *tmp
){
    if(args->solver_type == "jacobi"){
        calc_residual_cpu(sparse_mat, x_new, b, r, tmp, args->vec_size);
        *residual_norm = infty_vec_norm_cpu(r, args->vec_size);
    }

    if(args->solver_type == "gauss-seidel"){
        calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
        *residual_norm = infty_vec_norm_cpu(r, args->vec_size);
    }
    
    if(args->solver_type == "gmres"){
        // NOTE: While not needed for GMRES in theory, it is helpful to compare
        // a computed residual with g[-1] when debugging 
        calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
        for(int i = 0; i < args->vec_size; ++i){
            // std::cout << "outer_r[" << i << "] = "<< r[i] << std::endl;
        }
        *residual_norm = euclidean_vec_norm_cpu(r, args->vec_size);
#ifdef DEBUG_MODE
        std::cout << "computed residual_norm = " << *residual_norm << std::endl;
#endif
        // exit(0);
    }
    
    args->normed_residuals[args->loop_params->residual_count] = *residual_norm;
    
    // Only increment if we're still in the main solver loop
    if(flags->convergence_flag){
        // Don't increment
    }
    else if(args->loop_params->iter_count == args->loop_params->max_iters){
        // Don't increment
    }
    else if(*residual_norm < args->loop_params->stopping_criteria){
        // Don't increment
    }
    else{
        ++args->loop_params->residual_count;
    }
       
}

void iter_output(
    const double *x_approx,
    int N,
    int iter_count
){
    printf("On iter: %i, x appox is:\n", iter_count);
    for(int i = 0; i < N; ++i){
        printf("idx: %i, val: %f\n", i, x_approx[i]);
    }
}

void print_x(
    argType *args,
    double *x,
    double *x_new,
    double *x_old,
    int n_rows
){
    if(args->solver_type == "jacobi"){
        iter_output(x_new, args->vec_size, args->loop_params->iter_count);
    }
    if(args->solver_type == "gauss-seidel"){
        iter_output(x, args->vec_size, args->loop_params->iter_count);
    }
    if(args->solver_type == "gmres"){
        gmres_get_x(args->R, args->g, x, x_old, args->V, n_rows, args->restart_count, args->loop_params->iter_count, args->gmres_restart_len);
        iter_output(x, args->vec_size, args->loop_params->iter_count);
    }
}

void extract_largest_elems(
    const COOMtxData *coo_mat,
    std::vector<double> *largest_elems
){
    // #pragma omp parallel for schedule (static)
    // for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
    //     if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
    //         (*diag)[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
    //     }
    // }

    // #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        int row = coo_mat->I[nz_idx];
        // VT absValue = std::abs(coo_mat->values[nz_idx]);
        double absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

        // #pragma omp critical
        // {
            if (absValue > (*largest_elems)[row]) {
                (*largest_elems)[row] = absValue;
            // }
        }
    }
};

void scale_vector(    
    double *vec_to_scale,
    std::vector<double> *largest_elems,
    int vec_len
){
    #pragma omp parallel for schedule (static)
    for (int idx = 0; idx < vec_len; ++idx){
       vec_to_scale[idx] = vec_to_scale[idx] / (*largest_elems)[idx];
    }
};

void scale_matrix(
    COOMtxData *coo_mat,
    std::vector<double> *largest_elems
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        coo_mat->values[nz_idx] = coo_mat->values[nz_idx] / (*largest_elems)[coo_mat->I[nz_idx]];
    }
};

// TODO: MPI preprocessing will go here
void preprocessing(
    argType *args
){
    std::cout << "Preprocessing Matrix Data" << std::endl;
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
#else
    args->vec_size = args->coo_mat->n_cols;
#endif

#ifdef OUTPUT_SPARSITY
    std::string file_out_name = "output_matrix";
    std::cout << "Writing sparsity to mtx file..." << std::endl;
    args->coo_mat->write_to_mtx_file(0, file_out_name);
    exit(0);
    std::cout << "Finished" << std::endl;
#endif

    std::vector<double> largest_elems(args->vec_size, 0.0);
    extract_largest_elems(args->coo_mat, &largest_elems);

    // An optional, but usually necessary preprocessing step
    scale_matrix(args->coo_mat, &largest_elems);

    convert_to_crs(args->coo_mat, args->sparse_mat->crs_mat);
    
#ifdef __CUDACC__
    cudaMalloc(&(args->d_row_ptr), (args->sparse_mat->crs_mat->n_rows+1)*sizeof(int));
    cudaMalloc(&(args->d_col), (args->sparse_mat->crs_mat->nnz)*sizeof(int));
    cudaMalloc(&(args->d_val), (args->sparse_mat->crs_mat->nnz)*sizeof(double));
    cudaMemcpy(args->d_row_ptr, &(args->sparse_mat->crs_mat->row_ptr)[0], (args->sparse_mat->crs_mat->n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_col, &(args->sparse_mat->crs_mat->col)[0], (args->sparse_mat->crs_mat->nnz)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_val, &(args->sparse_mat->crs_mat->val)[0], (args->sparse_mat->crs_mat->nnz)*sizeof(double), cudaMemcpyHostToDevice);
#endif

    extract_diag(args->coo_mat, args->D);

    // Make b vector
    generate_vector(args->b, args->vec_size, args->flags->random_data, &(args->coo_mat->values)[0], args->loop_params->init_b);
    // ^ b should likely draw from A(min) to A(max) range of values
    scale_vector(args->b, &largest_elems, args->vec_size);

    // Make initial x vector
    generate_vector(args->x_old, args->vec_size, args->flags->random_data, &(args->coo_mat->values)[0], args->loop_params->init_x);
    scale_vector(args->x_old, &largest_elems, args->vec_size);

#ifdef USE_USPMV
    // Need to permute these vectors in accordance with SIGMA if using USpMV library
    double *D_perm = new double [args->vec_size];
    apply_permutation(D_perm, args->D, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
    // std::swap(D_perm, args->D);

    double *b_perm = new double [args->vec_size];
    apply_permutation(b_perm, args->b, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
    // std::swap(b_perm, args->b);

    // NOTE: Permuted w.r.t. columns due to symmetric permutation
    double *x_old_perm = new double[args->vec_size];
    apply_permutation(x_old_perm, args->x_old, &(args->sparse_mat->scs_mat->new_to_old_idx)[0], args->vec_size);
    // std::swap(x_old_perm, args->x_old);

    // Deep copy, so you can free memory
    // TODO: wrap in func
    for(int i = 0; i < args->vec_size; ++i){
        args->D[i] = D_perm[i];
        args->b[i] = b_perm[i];
        args->x_old[i] = x_old_perm[i];
    }

    delete D_perm;
    delete b_perm;
    delete x_old_perm;
#endif

#ifdef __CUDACC__
    // NOTE: Really only need to copy x_old, D, and b data, as all other host vectors are just zero at this point?
    cudaMemcpy(args->d_x_star, args->x_star, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_x_new, args->x_new, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_x_old, args->x_old, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_tmp, args->tmp, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_D, args->D, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_r, args->r, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(args->d_b, args->b, args->vec_size*sizeof(double), cudaMemcpyHostToDevice);
#endif

    // Precalculate stopping criteria
    // Easier to just do on the host for now
    calc_residual_cpu(args->sparse_mat, args->x_old, args->b, args->r, args->tmp, args->vec_size);

    if(args->solver_type == "gmres"){
        // args->beta = infty_vec_norm_cpu(args->r, args->vec_size);
        args->beta = euclidean_vec_norm_cpu(args->r, args->vec_size);
#ifdef DEBUG_MODE
        std::cout << "Beta = " << args->beta << std::endl;
#endif  

#ifdef DEBUG_MODE
    std::cout << "init_v = [";
        for(int i = 0; i < args->sparse_mat->crs_mat->n_rows; ++i){
            std::cout << args->init_v[i] << ", ";
        }
    std::cout << "]" << std::endl;
#endif
        scale(args->init_v, args->r, 1 / args->beta, args->vec_size);
#ifdef DEBUG_MODE
    std::cout << "init_v = [";
        for(int i = 0; i < args->sparse_mat->crs_mat->n_rows; ++i){
            std::cout << args->init_v[i] << ", ";
        }
    std::cout << "]" << std::endl;
#endif
    }

#ifdef DEBUG_MODE
    printf("initial residual = [");
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << args->r[i] << ",";
    }
    printf("]\n");
#endif

    double norm_r0;

    if(args->solver_type == "gmres"){
        norm_r0 = euclidean_vec_norm_cpu(args->r, args->vec_size);
    }
    else{
        norm_r0 = infty_vec_norm_cpu(args->r, args->vec_size);
    }
    args->loop_params->stopping_criteria = args->loop_params->tol * norm_r0; 



#ifdef DEBUG_MODE
    printf("norm(initial residual) = %f\n", norm_r0);
    // printf("stopping criteria = %f\n",args->loop_params->stopping_criteria);
    std::cout << "stopping criteria = " << args->loop_params->tol <<  " * " <<  norm_r0 << " = " << args->loop_params->stopping_criteria << std::endl;
#endif

// #ifdef __CUDACC__
//     // The first residual is computed on the host, and given to the device
//     // Easier to just do on the host for now, and give stopping criteria to device
//     cudaMalloc(&(args->loop_params->d_stopping_criteria), sizeof(double));
//     cudaMemcpy(args->loop_params->d_stopping_criteria, &(args->loop_params->stopping_criteria), sizeof(double), cudaMemcpyHostToDevice);
// #endif

}

void allocate_structs(
    argType *args
){
    std::cout << "Allocating space for general structs" << std::endl;
    double *x_star = new double[args->vec_size];
    double *x_old = new double[args->vec_size];
    double *x_new = new double[args->vec_size];
    double *tmp = new double[args->vec_size];
    double *D = new double[args->vec_size];
    double *r = new double[args->vec_size];
    double *b = new double[args->vec_size];
    double *normed_residuals = new double[args->loop_params->max_iters / args->loop_params->residual_check_len + 1];

    args->x_star = x_star;
    args->x_old = x_old;
    args->x_new = x_new;
    args->tmp = tmp;
    args->D = D;
    args->r = r;
    args->b = b;
    args->normed_residuals = normed_residuals;
}

void gmres_allocate_structs(
    argType *args
){
    std::cout << "Allocating space for GMRES structs" << std::endl;
    double *init_v = new double[args->vec_size];
    double *V = new double[args->vec_size * (args->loop_params->gmres_restart_len+1)]; // (m x n)
    double *H = new double[(args->loop_params->gmres_restart_len+1) * args->loop_params->gmres_restart_len]; // (m+1 x m) 
    double *H_tmp = new double[(args->loop_params->gmres_restart_len+1) * args->loop_params->gmres_restart_len]; // (m+1 x m)
    double *J = new double[(args->loop_params->gmres_restart_len+1) * (args->loop_params->gmres_restart_len+1)];
    double *R = new double[args->loop_params->gmres_restart_len * (args->loop_params->gmres_restart_len+1)]; // (m+1 x m)
    double *Q = new double[(args->loop_params->gmres_restart_len+1) * (args->loop_params->gmres_restart_len+1)]; // (m+1 x m+1)
    double *Q_copy = new double[(args->loop_params->gmres_restart_len+1) * (args->loop_params->gmres_restart_len+1)]; // (m+1 x m+1)
    double *g = new double[args->loop_params->gmres_restart_len+1];
    double *g_copy = new double[args->loop_params->gmres_restart_len+1];

    args->init_v = init_v;
    args->V = V;
    args->H = H;
    args->H_tmp = H_tmp;
    args->J = J;
    args->R = R;
    args->Q = Q;
    args->Q_copy = Q_copy;
    args->g = g;
    args->g_copy = g_copy;
    args->restart_count = 0;
}

#ifdef USE_SCAMAC
/* helper function:
 * split integer range [a...b-1] in n nearly equally sized pieces [ia...ib-1], for i=0,...,n-1 */
void split_range(ScamacIdx a, ScamacIdx b, ScamacIdx n, ScamacIdx i, ScamacIdx *ia, ScamacIdx *ib) {
  ScamacIdx m = (b-a-1)/n + 1;
  ScamacIdx d = n-(n*m -(b-a));
  if (i < d) {
    *ia = m*i + a;
    *ib = m*(i+1) + a;
  } else {
    *ia = m*d + (i-d)*(m-1) + a;
    *ib = m*d + (i-d+1)*(m-1) + a;
  }
}


void scamac_generate(
    argType *args,
    int* scamac_nrows,
    int* scamac_nnz,
    COOMtxData *mtx
){

/**  examples/MPI/ex_count_mpi.c
 *
 *   basic example:
 *   - read a matrix name/argument string from the command line
 *   - count the number of non-zeros, and compute the maximum norm (=max |entry|) and row-sum norm
 *
 *   Matrix rows are generated in parallel MPI processes.
 *   The ScamacGenerator and ScamacWorkspace is allocated per process.
 */
//   int mpi_world_size, mpi_rank;
//   MPI_Init(&argc, &argv);
//   MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);
//   MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  char *matargstr = args->scamac_args;
//   if (argc<=1) {
//     printf("usage: ex_count <matrix-argument string>\n\nexample: ex_count Hubbard\n         ex_count Hubbard,n_sites=14,n_fermions=8,U=1.3\n         ex_count TridiagonalReal,subdiag=0.5,supdiag=2\n");
//     my_mpi_error_handler();
//   }
//   matargstr=argv[1];

  ScamacErrorCode err;
  ScamacGenerator *my_gen;
  char *errstr = NULL;
    
  // set error handler for MPI (the only global ScaMaC variable!)
//   scamac_error_handler = my_mpi_error_handler;

  /* parse matrix name & parameters from command line to obtain a ScamacGenerator ... */
  /* an identical generator is created per MPI process */
  err = scamac_parse_argstr(matargstr, &my_gen, &errstr);
  /* ... and check for errors */
  if (err) {
    printf("-- Problem with matrix-argument string:\n-- %s\n---> Abort.\n",errstr);
    // my_mpi_error_handler();
  }
  
  /* check matrix parameters */
  err = scamac_generator_check(my_gen, &errstr);
  if (err) {
    printf("-- Problem with matrix parameters:\n-- %s---> Abort.\n",errstr);
    // my_mpi_error_handler();
  }
  
  /* finalize the generator ... */
  err=scamac_generator_finalize(my_gen);
  /* ... and check, whether the matrix dimension is too large */
  if (err==SCAMAC_EOVERFLOW) {
    // TODO: doesn't work with llvm
    // printf("-- matrix dimension exceeds max. IDX value (%"SCAMACPRIDX")\n---> Abort.\n",SCAMAC_IDX_MAX);
    // my_mpi_error_handler();
  }
  /* catch remaining errors */
  SCAMAC_CHKERR(err);
  
  /* query number of rows and max. number of non-zero entries per row */
  ScamacIdx nrow = scamac_generator_query_nrow(my_gen);
  ScamacIdx maxnzrow = scamac_generator_query_maxnzrow(my_gen);

//   double t1 = MPI_Wtime();

  /* ScamacWorkspace is allocated per MPI process */
  ScamacWorkspace * my_ws;
  SCAMAC_TRY(scamac_workspace_alloc(my_gen, &my_ws));

  /* allocate memory for column indices and values per MPI process*/
//   ScamacIdx *cind = malloc(maxnzrow * sizeof(long int));
  ScamacIdx *cind = new signed long int[maxnzrow];
  double *val;
  if (scamac_generator_query_valtype(my_gen) == SCAMAC_VAL_REAL) {
    // val = malloc(maxnzrow * sizeof *val);
    val = new double[maxnzrow];
  } else {
    /* valtype == SCAMAC_VAL_COMPLEX */
    // val = malloc(2*maxnzrow * sizeof(double));
    val = new double[maxnzrow];
  }

  ScamacIdx ia,ib;
  // this MPI process generates rows ia ... ib-1
  split_range(0,nrow, 1, 0, &ia, &ib);
  
  // allocate space
  int* scamac_rowPtr = new int[nrow + 1];
  int* scamac_col = new int[maxnzrow * nrow];
  double* scamac_val = new double[maxnzrow * nrow];

  // init counters
  int row_ptr_idx = 0;
  int scs_arr_idx = 0;
  scamac_rowPtr[0] = 0;

  for (ScamacIdx idx=ia; idx<ib; idx++) {
    ScamacIdx k;
    /* generate single row ... */
    SCAMAC_TRY(scamac_generate_row(my_gen, my_ws, idx, SCAMAC_DEFAULT, &k, cind, val));
    /* ... which has 0 <=k <= maxnzrow entries */

    // Assign SCAMAC arrays to scs array
    scamac_rowPtr[row_ptr_idx + 1] = scamac_rowPtr[row_ptr_idx] + k;
    for(int i = 0; i < k; ++i){
        scamac_col[scs_arr_idx] = cind[i]; // I dont know if these are "remade" every iteration, seems like it
        scamac_val[scs_arr_idx] = val[i];
        ++scs_arr_idx;
    }

    *scamac_nnz += k;
    ++row_ptr_idx;
  }
  *scamac_nrows = ib - ia;

        // Stupid to convert back to COO, only to convert back to scs. But safe for now.
    (mtx->I).resize(*scamac_nnz);
    (mtx->J).resize(*scamac_nnz);
    (mtx->values).resize(*scamac_nnz); 

    // for (int i = 0; i < *scamac_nrows + 1; ++i){
    //     std::cout << "scamac row ptr[" << i << "] = " << scamac_rowPtr[i] << std::endl;
    // }

    int elem_num = 0;
    for(int row = 0; row < *scamac_nrows; ++row){
        for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
            (mtx->I)[elem_num] = row;
            (mtx->J)[elem_num] = scamac_col[idx];
            (mtx->values)[elem_num] = scamac_val[idx];
            ++elem_num;
        }
    }

    // Verify everything is working as expected
    // for (int i = 0; i < *scamac_nrows + 1; ++i){
    //     std::cout << "scamac row ptr[" << i << "] = " << scamac_rowPtr[i] << std::endl;
    // }
    // for(int row = 0; row < *scamac_nrows; ++row){
    //     for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
    //         std::cout << "row = " << row << std::endl;
    //         std::cout << "scamac_col[" << idx << "] = " << scamac_col[idx] << std::endl;
    //         std::cout << "scamac_val[" << idx << "] = " << scamac_val[idx] << std::endl;
    //     }
    // }
        // for(int idx = 0; idx < (mtx->I).size(); ++idx){
        //     // for(int idx = scamac_rowPtr[row]; idx < scamac_rowPtr[row + 1]; ++idx){
        //         std::cout << "row[" << idx << "] = " << (mtx->I)[idx] << std::endl;
        //         std::cout << "col[" << idx << "] = " << (mtx->J)[idx] << std::endl;
        //         std::cout << "val[" << idx << "] = " << (mtx->values)[idx] << std::endl;
        //     }

  
  
  /* free local objects */
    delete[] scamac_rowPtr;
    delete[] scamac_col;
    delete[] scamac_val;
  free(cind);
  free(val);
  SCAMAC_TRY(scamac_workspace_free(my_ws));
  SCAMAC_TRY(scamac_generator_destroy(my_gen));
}

void scamac_make_mtx(
    argType *args,
    COOMtxData *coo_mat
){
    int scamac_nrows = 0;
    int scamac_nnz = 0;

    // Fill scs arrays with proper data
    scamac_generate(
        args, 
        &scamac_nrows,
        &scamac_nnz,
        coo_mat
    );
    
    // Finish up mtx struct creation (TODO: why do I do it this way?)
    coo_mat->n_rows = (std::set<int>( (coo_mat->I).begin(), (coo_mat->I).end() )).size();
    coo_mat->n_cols = (std::set<int>( (coo_mat->J).begin(), (coo_mat->J).end() )).size();
    coo_mat->nnz = (coo_mat->values).size();
}
#endif

#ifdef __CUDACC__

void gpu_allocate_structs(
    argType *args
){
    cudaMalloc(&(args->d_x_star), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_new), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_old), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_tmp), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_D), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_r), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_b), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_normed_residuals), (args->loop_params->max_iters / args->loop_params->residual_check_len + 1)*sizeof(double));
}

#endif