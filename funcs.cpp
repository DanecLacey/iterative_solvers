#include <stdio.h>
#include <iostream>
#include <random>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <math.h>
#include <sys/time.h>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <cmath>
#include <vector>


#include "structs.hpp"
#include "kernels.hpp"
#include "mmio.h"

void assign_cli_inputs(
    int argc,
    char *argv[],
    std::string *matrix_file_name,
    std::string *solver_type
    )
{
    if(argc != 3){
        printf("ERROR: assign_cli_inputs: Please only select a .mtx file name and solver type [-j (Jacobi) / -gs (Gauss-Seidel) / -t (Trivial)].\n");
        exit(1);
    }

    *matrix_file_name = argv[1];
    std::string st = argv[2];

    if(st == "-j"){
        *solver_type = "jacobi"; 
    }
    else if(st == "-gs"){
        *solver_type = "gauss-seidel";
    }
    else if(st == "-t"){
        *solver_type = "trivial";
    }
    else{
        printf("ERROR: assign_cli_inputs: Please choose an available solver type [-j (Jacobi) / -gs (Gauss-Seidel) / -t (Trivial)].\n");
        exit(1);
    }
}

inline void sort_perm(int *arr, int *perm, int len, bool rev=false)
{
    if(rev == false) {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] < arr[b]); });
    } else {
        std::stable_sort(perm+0, perm+len, [&](const int& a, const int& b) {return (arr[a] > arr[b]); });
    }
}

void read_mtx(
    const std::string matrix_file_name,
    COOMtxData *full_coo_mtx
    )
{
    char* filename = const_cast<char*>(matrix_file_name.c_str());
    int nrows, ncols, nnz;
    double *val_ptr;
    int *I_ptr;
    int *J_ptr;

    MM_typecode matcode;
    FILE *f;

    if ((f = fopen(filename, "r")) == NULL) {printf("Unable to open file");}

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("mm_read_unsymetric: Could not process Matrix Market banner ");
        printf(" in file [%s]\n", filename);
        // return -1;
    }

    fclose(f);

    // bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool compatible_flag = (mm_is_sparse(matcode) && (mm_is_real(matcode)||mm_is_pattern(matcode)||mm_is_integer(matcode))) && (mm_is_symmetric(matcode) || mm_is_general(matcode));
    bool symm_flag = mm_is_symmetric(matcode);
    bool pattern_flag = mm_is_pattern(matcode);

    if(!compatible_flag)
    {
        printf("The matrix market file provided is not supported.\n Reason :\n");
        if(!mm_is_sparse(matcode))
        {
            printf(" * matrix has to be sparse\n");
        }

        if(!mm_is_real(matcode) && !(mm_is_pattern(matcode)))
        {
            printf(" * matrix has to be real or pattern\n");
        }

        if(!mm_is_symmetric(matcode) && !mm_is_general(matcode))
        {
            printf(" * matrix has to be einther general or symmetric\n");
        }

        exit(0);
    }

    //int ncols;
    int *row_unsorted;
    int *col_unsorted;
    double *val_unsorted;

    if(mm_read_unsymmetric_sparse<double, int>(filename, &nrows, &ncols, &nnz, &val_unsorted, &row_unsorted, &col_unsorted) < 0)
    {
        printf("Error in file reading\n");
        exit(1);
    }
    if(nrows != ncols)
    {
        printf("Matrix not square. Currently only square matrices are supported\n");
        exit(1);
    }

    //If matrix market file is symmetric; create a general one out of it
    if(symm_flag)
    {
        // printf("Creating a general matrix out of a symmetric one\n");

        int ctr = 0;

        //this is needed since diagonals might be missing in some cases
        for(int idx=0; idx<nnz; ++idx)
        {
            ++ctr;
            if(row_unsorted[idx]!=col_unsorted[idx])
            {
                ++ctr;
            }
        }

        int new_nnz = ctr;

        int *row_general = new int[new_nnz];
        int *col_general = new int[new_nnz];
        double *val_general = new double[new_nnz];

        int idx_gen=0;

        for(int idx=0; idx<nnz; ++idx)
        {
            row_general[idx_gen] = row_unsorted[idx];
            col_general[idx_gen] = col_unsorted[idx];
            val_general[idx_gen] = val_unsorted[idx];
            ++idx_gen;

            if(row_unsorted[idx] != col_unsorted[idx])
            {
                row_general[idx_gen] = col_unsorted[idx];
                col_general[idx_gen] = row_unsorted[idx];
                val_general[idx_gen] = val_unsorted[idx];
                ++idx_gen;
            }
        }

        free(row_unsorted);
        free(col_unsorted);
        free(val_unsorted);

        nnz = new_nnz;

        //assign right pointers for further proccesing
        row_unsorted = row_general;
        col_unsorted = col_general;
        val_unsorted = val_general;

        // delete[] row_general;
        // delete[] col_general;
        // delete[] val_general;
    }

    //permute the col and val according to row
    int *perm = new int[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        perm[idx] = idx;
    }

    sort_perm(row_unsorted, perm, nnz);

    int *col = new int[nnz];
    int *row = new int[nnz];
    double *val = new double[nnz];

    // pramga omp parallel for?
    for(int idx=0; idx<nnz; ++idx)
    {
        col[idx] = col_unsorted[perm[idx]];
        val[idx] = val_unsorted[perm[idx]];
        row[idx] = row_unsorted[perm[idx]];
    }

    delete[] perm;
    delete[] col_unsorted;
    delete[] val_unsorted;
    delete[] row_unsorted;

    full_coo_mtx->values = std::vector<double>(val, val + nnz);
    full_coo_mtx->I = std::vector<int>(row, row + nnz);
    full_coo_mtx->J = std::vector<int>(col, col + nnz);
    full_coo_mtx->n_rows = nrows;
    full_coo_mtx->n_cols = ncols;
    full_coo_mtx->nnz = nnz;
    full_coo_mtx->is_sorted = 1; // TODO: not sure
    full_coo_mtx->is_symmetric = 0; // TODO: not sure

    delete[] val;
    delete[] row;
    delete[] col;
}

// TODO
//void convert_coo_to_crs(
//        )
//{
//    
//}

void split_upper_lower_diagonal(
    COOMtxData *full_coo_mtx,
    COOMtxData *U_coo_mtx,
    COOMtxData *L_coo_mtx,
    std::vector<double> *D_coo_vec
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
            // Copy element to vector representing diagonal matrix
            // NOTE: Don't need push_back because we know the size
            if(abs(full_coo_mtx->values[nz_idx]) < 1e-15 && !explitit_zero_warning_flag){ // NOTE: error tolerance too tight?
                printf("WARNING: split_upper_lower_diagonal: explicit zero detected on diagonal at nz_idx %i.\n"
                        "row: %i, col: %i, val: %f.\n", nz_idx, full_coo_mtx->I[nz_idx], full_coo_mtx->J[nz_idx], full_coo_mtx->values[nz_idx]);
                explitit_zero_warning_flag = true;
            }
            (*D_coo_vec)[D_coo_vec_count] = full_coo_mtx->values[nz_idx];
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

void generate_vector(
    std::vector<double> *vec_to_populate,
    int size,
    bool rand_flag,
    int initial_val
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

double infty_mtx_coo_norm(
    COOMtxData *full_coo_mtx
){
    // Accumulate with sum all the elements in each row
    std::vector<double> row_sums(full_coo_mtx->n_cols, 0);

    // Good OpenMP candidate
    for(int nz_idx = 0; nz_idx < full_coo_mtx->nnz; ++nz_idx){
        row_sums[full_coo_mtx->I[nz_idx]] += abs(full_coo_mtx->values[nz_idx]); 
    }

    // The largest sum is the matrix infty norm
    return infty_vec_norm(&row_sums);
}

/* Residual here is the distance from A*x_new to b, where the norm
 is the infinity norm: ||A*x_new-b||_infty */
double calc_residual(
    COOMtxData *full_coo_mtx,
    std::vector<double> *x_new,
    std::vector<double> *b
){
    std::vector<double> A_x_new(full_coo_mtx->n_cols); 
    mtx_spmv_coo(&A_x_new, full_coo_mtx, x_new);

    std::vector<double> A_x_new_minus_b(full_coo_mtx->n_cols);
    std::vector<double> neg_b(full_coo_mtx->n_cols);
    neg_b = *b;
    std::transform( // negate
        neg_b.cbegin(), 
        neg_b.cend(), 
        neg_b.begin(), 
        std::negate<double>()
    );
    sum_vectors(&A_x_new_minus_b, &A_x_new, &neg_b);

    return infty_vec_norm(&A_x_new_minus_b);
}

void residuals_output(
    bool print_residuals,
    std::vector<double> *residuals_vec,
    int iter_count
){
    for(int i = 0; i < iter_count; ++i){
        std::cout << "||A*x_" << i << " - b||_infty = " << std::setprecision(16) << (*residuals_vec)[i] << std::endl;
    }
}

void summary_output(
    COOMtxData *full_coo_mtx,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *residuals_vec,
    std::string *solver_type,
    int max_iters,
    bool convergence_flag,
    bool print_residuals,
    int iter_count,
    double total_time_elapsed,
    double calc_time_elapsed,
    double tol
){
    if(convergence_flag){
        // x_new ~ A^{-1}b
        std::cout << "\n" << *solver_type << " solver converged in: " << iter_count << " iterations." << std::endl;
    }
    else{
        // x_new !~ A^{-1}b
        std::cout << "\n" << *solver_type << " solver did not converge after " << max_iters << " iterations." << std::endl;
    }
    std::cout << "The residual of the final iteration is: ||A*x_star - b||_infty = " <<
    std::setprecision(16) << (*residuals_vec)[iter_count] << ".\n";
    std::cout << "The total elapsed time was: " << total_time_elapsed << "[s]." << std::endl;
    std::cout << "Out of which, the pre-processing time was: " << total_time_elapsed - calc_time_elapsed <<
    "[s], and the computation time was: " << calc_time_elapsed << "[s]." <<std::endl;

    if(print_residuals){
        residuals_output(print_residuals, residuals_vec, iter_count);
    }
}

void iter_output(
    std::vector<double> *x_approx,
    int iter_count
){
    printf("On iter: %i, x appox is:\n", iter_count);
    for(int i = 0; i < x_approx->size(); ++i){
        printf("idx: %i, val: %f\n", i, (*x_approx)[i]);
    }
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

void jacobi_iteration(
    COOMtxData *full_coo_L_plus_U_mtx,
    std::vector<double> *D_inv_coo_vec,
    std::vector<double> *neg_D_inv_coo_vec,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
){
    // TODO: definetly dont initialize these in-loop, bad for performance and stupid
    std::vector<double> L_plus_U_mult_x_old(D_inv_coo_vec->size(), 0);
    std::vector<double> neg_D_inv_mult_L_plus_U_mult_x_old(D_inv_coo_vec->size(), 0);
    std::vector<double> D_inv_mult_b(D_inv_coo_vec->size(), 0);
    
    // 1. SpMV (L + U)*x_old
    mtx_spmv_coo(&L_plus_U_mult_x_old, full_coo_L_plus_U_mtx, x_old);

    // 2. "SpMV" D^{-1}*b (elementwise multiplication) //NOTE: 1 and 2 independent
    vec_spmv_coo(&D_inv_mult_b, D_inv_coo_vec, b);

    // 3 "SpMV" -D^{-1}*((L + U)*x_old) (elementwise multiplication)
    vec_spmv_coo(&neg_D_inv_mult_L_plus_U_mult_x_old, neg_D_inv_coo_vec, &L_plus_U_mult_x_old);

    // 4. Vector Sum x_new = -D^{-1}*(L + U)*x_old + D^{-1}*b (elementwise sum)
    sum_vectors(x_new, &neg_D_inv_mult_L_plus_U_mult_x_old, &D_inv_mult_b);
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

// TODO: move solvers to a seperate file
void jacobi_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    COOMtxData *full_coo_mtx,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
)
{
    std::cout << "Do I get into Jacobi Solve?" << std::endl;
    int ncols = full_coo_mtx->n_cols;
    double residual;
    double stopping_criteria;

    // Declare structs
    std::vector<double> D_coo_vec(ncols);
    COOMtxData L_coo_mtx, U_coo_mtx, full_coo_L_plus_U_mtx;

    // Split COO matrix 
    split_upper_lower_diagonal(full_coo_mtx, &U_coo_mtx, &L_coo_mtx, &D_coo_vec);

    // NOTE: The following computations can be performed outside of the main loop, since they are not changing
    // Form L+U matrix-matrix elementwise sum 
    sum_matrices(&full_coo_L_plus_U_mtx, &U_coo_mtx, &L_coo_mtx); 

    // Form -D^{-1}
    std::vector<double> neg_D_inv_coo_vec(ncols);
    std::vector<double> D_inv_coo_vec(ncols);
    gen_neg_inv(&neg_D_inv_coo_vec, &D_inv_coo_vec, &D_coo_vec);
    
    if(flags->print_iters){
        iter_output(x_old, loop_params->iter_count);
        printf("\n");
    }
    // std::cout << "x_old vector:" << std::endl;
    // for(int i = 0; i < full_coo_mtx.n_cols; ++i){
    //     std::cout << x_old[i] << std::endl;
    // }

    // Precalculate what we can of stopping criteria
    double infty_norm_A = infty_mtx_coo_norm(full_coo_mtx);
    double infty_norm_b = infty_vec_norm(b);

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    // Perform Jacobi iterations until error cond. satisfied
    // Using relation: x_new = -D^{-1}*(L + U)*x_old + D^{-1}*b
    // After Jacobi iteration loop, x_new ~ A^{-1}b
    (*residuals_vec)[0] = calc_residual(full_coo_mtx, x_old, b);

    std::cout << "The initial residual is: " << (*residuals_vec)[0] << std::endl;


    // Tasking candidate
    do{
        std::cout << "Do I get into The do loop?" << std::endl;
        jacobi_iteration(&full_coo_L_plus_U_mtx, &D_inv_coo_vec, &neg_D_inv_coo_vec, b, x_old, x_new);
        ++loop_params->iter_count;
        residual = calc_residual(full_coo_mtx, x_new, b);
        stopping_criteria = loop_params->tol * (infty_norm_A * infty_vec_norm(x_new) + infty_norm_b);
        if(flags->print_residuals){
            (*residuals_vec)[loop_params->iter_count] = residual;
        }
        if(flags->print_iters){
            iter_output(x_new, loop_params->iter_count);
            // printf("The residual is %f.\n\n", residual);
        }  
        if(loop_params->iter_count >= loop_params->max_iters){
            flags->convergence_flag = false;
            break;
        }

        std::cout << "[";
        for(int i = 0; i < x_new->size(); ++i){
            std::cout << (*x_new)[i] << ", ";
        }
        std::cout << "]" << std::endl;
        std::swap(*x_new, *x_old);
        std::cout << "residual: " << residual << std::endl;
        std::cout << "stopping_criteria: " << stopping_criteria << std::endl; 
        
    } while(residual > stopping_criteria);

    std::swap(*x_old, *x_star);

    // End timer
    (*calc_time_elapsed) = end_time(&calc_time_start, &calc_time_end);
}

void gs_iteration(
    COOMtxData *full_coo_L_plus_U_mtx,
    std::vector<double> *D_inv_coo_vec,
    std::vector<double> *neg_D_inv_coo_vec,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
){}

void gs_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    COOMtxData *full_coo_mtx,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){}

void trivial_iteration(
    COOMtxData *full_coo_L_plus_U_mtx,
    std::vector<double> *D_inv_coo_vec,
    std::vector<double> *neg_D_inv_coo_vec,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
){}

void trivial_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    COOMtxData *full_coo_mtx,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){}

void FOM_iteration(
    COOMtxData *full_coo_L_plus_U_mtx,
    std::vector<double> *D_inv_coo_vec,
    std::vector<double> *neg_D_inv_coo_vec,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
){}

void initialize_with_zeros_COO(
    COOMtxData *full_coo_mtx,
    int nrows,
    int ncols
){
    full_coo_mtx->n_rows = nrows;
    // int n_rows{};

    full_coo_mtx->n_cols = ncols;
    // int n_cols{};

    full_coo_mtx->nnz = 0;
    // int nnz{};

    full_coo_mtx->is_sorted = true;
    // bool is_sorted{}; = 1;

    full_coo_mtx->is_symmetric = false;
    // bool is_symmetric{}; = 1;

    
    std::vector<double> sized_zero_vec(nrows*ncols, 0.0);

    full_coo_mtx->I = ;
    // std::vector<int> J;

    full_coo_mtx->J = ;
    // std::vector<int> J;

    full_coo_mtx->values = ;
    // std::vector<double> values;

}

void FOM_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    COOMtxData *full_coo_mtx,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){
    // This does not seem like an"iterative" method
    int ncols = full_coo_mtx->n_cols;
    double residual;

    // Declare structs
    std::vector<double> w_new(ncols);
    std::vector<double> w_old(ncols);
    COOMtxData V_m, H_m;

    // Just something for now
    int target_dim = ncols - .25 * ncols

    initialize_with_zeros_COO(&V_m, target_dim, target_dim);
    initialize_with_zeros_COO(&H_m, target_dim, target_dim);

    

    FOM_iteration();
}

void Arnoldi_iteration(
    std::vector<double> *v_1,
    COOMtxData *H_m,
    COOMtxData *V_m,
    int target_dim,
    Flags *flags,
    LoopParams *loop_params
){}

void write_residuals_to_file(std::vector<double> *residuals_vec){
    std::fstream out_file;
    out_file.open("residuals.txt", std::fstream::out | std::fstream::trunc);

    for(int i = 0; i < residuals_vec->size(); ++i){
        out_file << i << " " << (*residuals_vec)[i] << "\n";
    }
    out_file.close();
}