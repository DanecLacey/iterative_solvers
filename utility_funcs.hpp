#ifndef UTILITY_FUNCS_H
#define UTILITY_FUNCS_H
#include <stdio.h>
#include <iostream>
#include <random>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <math.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <iomanip>
#include <set>
#include <omp.h>

#ifdef USE_EIGEN
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#endif

#ifdef USE_SCAMAC
#include "scamac.h"
#endif

#include "kernels.hpp"
#include "mmio.h"
#include "sparse_matrix.hpp"
#include "structs.hpp"

#define xstr(s) str(s)
#define str(s) #s

template <typename VT>
void init(
    VT *vec,
    VT val,
    long size
){
    // TODO: validate first touch policy?
    #pragma omp parallel for
    for(int i = 0; i < size; ++i){
        vec[i] = val;
    }
}

template <typename VT>
void init_identity(
    VT *mat,
    VT val,
    int n_rows,
    int n_cols
){
    // TODO: validate first touch policy?
    #pragma omp parallel for
    for(int i = 0; i < n_rows; ++i){
        for(int j = 0; j < n_cols; ++j){
            if(i == j){
                mat[n_cols*i + j] = 1.0;
            }
            else{
                mat[n_cols*i + j] = 0.0;
            }
        }
    }
}

template <typename VT>
void generate_vector(
    VT *vec_to_populate,
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

template <typename VT>
void extract_diag(
    const COOMtxData<double> *coo_mat,
    VT *diag,
    bool take_sqrt = false
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        if(coo_mat->I[nz_idx] == coo_mat->J[nz_idx]){
            if(take_sqrt){
                diag[coo_mat->I[nz_idx]] = std::sqrt(std::abs(coo_mat->values[nz_idx]));
            }
            else{
                diag[coo_mat->I[nz_idx]] = coo_mat->values[nz_idx];
            }
        }
    }
}

template <typename VT>
void compare_with_direct(
    CRSMtxData<VT> *crs_mat,
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

template <typename VT, typename DT>
void split_L_U(
    bool *full_coo_mtx_is_sorted,
    bool *full_coo_mtx_is_symmetric,
    long *full_coo_mtx_n_cols,
    long *full_coo_mtx_n_rows,
    long *full_coo_mtx_nnz,
    std::vector<int> *full_coo_mtx_I,
    std::vector<int> *full_coo_mtx_J,
    std::vector<VT> *full_coo_mtx_values,
    bool *L_coo_mtx_is_sorted,
    bool *L_coo_mtx_is_symmetric,
    long *L_coo_mtx_n_cols,
    long *L_coo_mtx_n_rows,
    long *L_coo_mtx_nnz,
    std::vector<int> *L_coo_mtx_I,
    std::vector<int> *L_coo_mtx_J,
    std::vector<DT> *L_coo_mtx_values,
    bool *U_coo_mtx_is_sorted,
    bool *U_coo_mtx_is_symmetric,
    long *U_coo_mtx_n_cols,
    long *U_coo_mtx_n_rows,
    long *U_coo_mtx_nnz,
    std::vector<int> *U_coo_mtx_I,
    std::vector<int> *U_coo_mtx_J,
    std::vector<DT> *U_coo_mtx_values
){
    bool explitit_zero_warning_flag = false;
    int L_coo_mtx_count = 0;
    int U_coo_mtx_count = 0;
    int D_coo_vec_count = 0;

    // Force same dimensions for consistency
    *U_coo_mtx_n_rows = *full_coo_mtx_n_rows;
    *U_coo_mtx_n_cols = *full_coo_mtx_n_cols;
    *U_coo_mtx_is_sorted = *full_coo_mtx_is_sorted;
    *U_coo_mtx_is_symmetric = false;
    *L_coo_mtx_n_rows = *full_coo_mtx_n_rows;
    *L_coo_mtx_n_cols = *full_coo_mtx_n_cols;
    *L_coo_mtx_is_sorted = *full_coo_mtx_is_sorted;
    *L_coo_mtx_is_symmetric = false;

    for(int nz_idx = 0; nz_idx < *full_coo_mtx_nnz; ++nz_idx){
        // If column and row less than i, this nz is in the L matrix
        if((*full_coo_mtx_J)[nz_idx] < (*full_coo_mtx_I)[nz_idx]){
            // Copy element to lower matrix
            L_coo_mtx_I->push_back((*full_coo_mtx_I)[nz_idx]);
            L_coo_mtx_J->push_back((*full_coo_mtx_J)[nz_idx]);
            L_coo_mtx_values->push_back((*full_coo_mtx_values)[nz_idx]);
            ++(*L_coo_mtx_nnz);
            // std::cout << full_coo_mtx->values[nz_idx] << " sent to lower matrix" << std::endl;
        }
        else if((*full_coo_mtx_J)[nz_idx] > (*full_coo_mtx_I)[nz_idx]){
            // Copy element to upper matrix
            U_coo_mtx_I->push_back((*full_coo_mtx_I)[nz_idx]);
            U_coo_mtx_J->push_back((*full_coo_mtx_J)[nz_idx]);
            U_coo_mtx_values->push_back((*full_coo_mtx_values)[nz_idx]);
            ++(*U_coo_mtx_nnz);
            // std::cout << full_coo_mtx->values[nz_idx] << " sent to upper matrix" << std::endl;
        }
        else if((*full_coo_mtx_I)[nz_idx] == (*full_coo_mtx_J)[nz_idx]){
        //     // Copy element to vector representing diagonal matrix
        //     // NOTE: Don't need push_back because we know the size
            if(std::abs(static_cast<double>((*full_coo_mtx_values)[nz_idx])) < 1e-15 && !explitit_zero_warning_flag){ // NOTE: error tolerance too tight?
                printf("WARNING: split_upper_lower_diagonal: explicit zero detected on diagonal at nz_idx %i.\n"
                        "row: %i, col: %i, val: %f.\n", nz_idx, (*full_coo_mtx_I)[nz_idx], (*full_coo_mtx_J)[nz_idx], (*full_coo_mtx_values)[nz_idx]);
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
    int copied_elems_count = *L_coo_mtx_nnz + *U_coo_mtx_nnz + D_coo_vec_count; 
    if(copied_elems_count != *full_coo_mtx_nnz){
        printf("ERROR: split_upper_lower_diagonal: only %i out of %i elements were copied from full_coo_mtx.\n", copied_elems_count, *full_coo_mtx_nnz);
        exit(1);
    }
}

template<typename COOT, typename VT>
void convert_to_crs(
    long *coo_mat_n_rows,
    long *coo_mat_n_cols,
    long *coo_mat_nnz,
    std::vector<int> *coo_mat_I,
    std::vector<int> *coo_mat_J,
    std::vector<COOT> *coo_mat_values,
    CRSMtxData<VT> *crs_mat
    )
{
    crs_mat->n_rows = *coo_mat_n_rows;
    crs_mat->n_cols = *coo_mat_n_cols;
    crs_mat->nnz = *coo_mat_nnz;

    crs_mat->row_ptr = new int[crs_mat->n_rows+1];
    int *nnzPerRow = new int[crs_mat->n_rows];

    crs_mat->col = new int[crs_mat->nnz];
    crs_mat->val = new VT[crs_mat->nnz];

    for(int idx = 0; idx < crs_mat->nnz; ++idx)
    {
        crs_mat->col[idx] = (*coo_mat_J)[idx];
        crs_mat->val[idx] = (*coo_mat_values)[idx];
    }

    for(int i = 0; i < crs_mat->n_rows; ++i)
    { 
        nnzPerRow[i] = 0;
    }

    //count nnz per row
    for(int i=0; i < crs_mat->nnz; ++i)
    {
        ++nnzPerRow[(*coo_mat_I)[i]];
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

template <typename VT>
void record_residual_norm(
    argType<VT> *args,
    Flags *flags,
    SparseMtxFormat<VT> *sparse_mat,
    VT *r,
    VT *x,
    VT *b,
    VT *x_new,
    VT *tmp,
    VT *tmp_perm,
#ifdef USE_AP
    double *x_dp,
    double *x_new_dp,
    double *tmp_dp,
    double *tmp_perm_dp,
    float *x_sp,
    float *x_new_sp,
    float *tmp_sp,
    float *tmp_perm_sp,
#ifdef HAVE_HALF_MATH
    _Float16 *x_hp,
    _Float16 *x_new_hp,
    _Float16 *tmp_hp,
    _Float16 *tmp_perm_hp,
#endif
#endif
    double *residual_norm
){
    if(args->solver_type == "jacobi"){
#ifdef USE_AP
        std::string working_precision = xstr(WORKING_PRECISION);
        if(working_precision == "double"){
            calc_residual_cpu<VT, double>(sparse_mat, x_new_dp, b, r, tmp_dp, tmp_perm_dp, args->coo_mat->n_cols);
        }
        else if(working_precision == "float"){
            calc_residual_cpu<VT, float>(sparse_mat, x_new_sp, b, r, tmp_sp, tmp_perm_sp, args->coo_mat->n_cols);
        }
        else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
            calc_residual_cpu<VT, _Float16>(sparse_mat, x_new_hp, b, r, tmp_hp, tmp_perm_hp, args->coo_mat->n_cols);
#endif
        }
#else
        calc_residual_cpu<VT, VT>(sparse_mat, x_new, b, r, tmp, tmp_perm, args->coo_mat->n_cols);
#endif
        *residual_norm = infty_vec_norm_cpu<VT>(r, args->coo_mat->n_cols);
    }
    else if(args->solver_type == "gauss-seidel"){
#ifdef USE_AP
        std::string working_precision = xstr(WORKING_PRECISION);
        if(working_precision == "double"){
            calc_residual_cpu<VT, double>(sparse_mat, x_dp, b, r, tmp_dp, tmp_perm_dp, args->coo_mat->n_cols);
        }
        else if(working_precision == "float"){
            calc_residual_cpu<VT, float>(sparse_mat, x_sp, b, r, tmp_sp, tmp_perm_sp, args->coo_mat->n_cols);
        }
        else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
            calc_residual_cpu<VT, _Float16>(sparse_mat, x_hp, b, r, tmp_hp, tmp_perm_hp, args->coo_mat->n_cols);
#endif
        }

#else
        calc_residual_cpu<VT, VT>(sparse_mat, x, b, r, tmp, tmp_perm, args->coo_mat->n_cols);
#endif
        *residual_norm = infty_vec_norm_cpu<VT>(r, args->coo_mat->n_cols);
    }
    else if(args->solver_type == "gmres"){
        // NOTE: While not needed for GMRES in theory, it is helpful to compare
        // a computed residual with g[-1] when debugging 
//         calc_residual_cpu(sparse_mat, x, b, r, tmp, args->vec_size);
//         *residual_norm = euclidean_vec_norm_cpu(r, args->vec_size);
// #ifdef DEBUG_MODE
//         std::cout << "computed residual_norm = " << *residual_norm << std::endl;
// #endif

        // The residual norm is already implicitly computed, and is output from the GMRES iteration
    }
    else if(args->solver_type == "conjugate-gradient"){
        // Not necessary since CG computes the residual vector within the algorithm.
        // *residual_norm = infty_vec_norm_cpu<VT>(r, args->coo_mat->n_cols);
        *residual_norm = euclidean_vec_norm_cpu<VT>(r, args->coo_mat->n_cols);

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

template <typename VT>
void iter_output(
    const VT *x_approx,
    int N,
    int iter_count
){
    printf("On iter: %i, x appox is:\n", iter_count);
    for(int i = 0; i < N; ++i){
        printf("idx: %i, val: %f\n", i, x_approx[i]);
    }
}

template <typename VT>
void scale_vector(    
    VT *vec_to_scale,
    std::vector<double> *largest_elems,
    int vec_len
){
    #pragma omp parallel for schedule (static)
    for (int idx = 0; idx < vec_len; ++idx){
       vec_to_scale[idx] = vec_to_scale[idx] / ((*largest_elems)[idx] * (*largest_elems)[idx]);
    //    vec_to_scale[idx] = vec_to_scale[idx] / (*largest_elems)[idx];

    }
};

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

template <typename VT>
void scamac_generate(
    argType<VT> *args,
    int* scamac_nrows,
    int* scamac_nnz,
    COOMtxData<double> *mtx
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

// NOTE: This will always make double precision?
template <typename VT>
void scamac_make_mtx(
    argType<VT> *args,
    COOMtxData<double> *coo_mat
){
    int scamac_nrows = 0;
    int scamac_nnz = 0;

    // Fill scs arrays with proper data
    scamac_generate<VT>(
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

void bogus_init_pin(void){
    
    // Just to take overhead of pinning away from timers
    int num_threads;
    double bogus = 0.0;

    #pragma omp parallel
    {
        num_threads = omp_get_num_threads();
    }

    #pragma omp parallel for
    for(int i = 0; i < num_threads; ++i){
        bogus += 1;
    }

    if(bogus < 100){
        printf("");
    }
}

template <typename VT>
void extract_largest_row_elems(
    const COOMtxData<VT> *coo_mat,
    std::vector<double> *largest_row_elems
){
    // #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        int row = coo_mat->I[nz_idx];
        // VT absValue = std::abs(coo_mat->values[nz_idx]);
        double absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

        // #pragma omp critical
        // {
            if (absValue > (*largest_row_elems)[row]) {
                (*largest_row_elems)[row] = absValue;
            // }
        }
    }
};

template <typename VT>
void extract_largest_col_elems(
    const COOMtxData<VT> *coo_mat,
    std::vector<double> *largest_col_elems
){
    // #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        int col = coo_mat->J[nz_idx];
        // VT absValue = std::abs(coo_mat->values[nz_idx]);
        double absValue = std::abs(static_cast<double>(coo_mat->values[nz_idx]));

        // #pragma omp critical
        // {
            if (absValue > (*largest_col_elems)[col]) {
                (*largest_col_elems)[col] = absValue;
            // }
        }
    }
};

template <typename VT>
void scale_matrix_rows(
    COOMtxData<VT> *coo_mat,
    std::vector<double> *largest_row_elems
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        coo_mat->values[nz_idx] = coo_mat->values[nz_idx] / (*largest_row_elems)[coo_mat->I[nz_idx]];
    }
};

template <typename VT>
void scale_matrix_cols(
    COOMtxData<VT> *coo_mat,
    std::vector<double> *largest_col_elems
){
    #pragma omp parallel for schedule (static)
    for (int nz_idx = 0; nz_idx < coo_mat->nnz; ++nz_idx){
        coo_mat->values[nz_idx] = coo_mat->values[nz_idx] / (*largest_col_elems)[coo_mat->J[nz_idx]];
    }
};

template <typename VT>
void equilibrate_matrix(COOMtxData<VT> *coo_mat){
    std::vector<double> largest_row_elems(coo_mat->n_cols, 0.0);
    extract_largest_row_elems(coo_mat, &largest_row_elems);
    scale_matrix_rows(coo_mat, &largest_row_elems);

    std::vector<double> largest_col_elems(coo_mat->n_cols, 0.0);
    extract_largest_col_elems(coo_mat, &largest_col_elems);
    scale_matrix_cols(coo_mat, &largest_col_elems);
}

#ifdef USE_LIKWID
    void register_likwid_markers(){
        #pragma omp parallel
        {
#ifdef USE_USPMV
#ifdef USE_AP
            // LIKWID_MARKER_REGISTER("uspmv_ap_crs_benchmark");
            LIKWID_MARKER_REGISTER("uspmv_ap_scs_benchmark");
#else
            LIKWID_MARKER_REGISTER("uspmv_crs_benchmark");
#endif

#else 
            LIKWID_MARKER_REGISTER("native_spmv_benchmark");
#endif
        }
    }
#endif
#endif /*UTILITY_FUNCS_H*/
