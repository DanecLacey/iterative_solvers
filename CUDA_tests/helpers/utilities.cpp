#include "structs.hpp"
#include <stdio.h>
#include <time.h>
#include <limits>
#include <mkl.h>

void generate_x_and_y(
    double *x,
    double *y,
    int N,
    int nnz,
    bool rand_flag,
    double *values,
    double initial_val
){
    if(rand_flag){
        double upper_bound = std::numeric_limits<double>::min();
        double lower_bound = std::numeric_limits<double>::max();
        for(int i = 0; i < nnz; ++i){
            if(values[i] > upper_bound) upper_bound = values[i];
            if(values[i] < lower_bound) lower_bound = values[i];
        }
        srand(time(nullptr));

        double range = (upper_bound - lower_bound); 
        double div = RAND_MAX / range;

        for(int i = 0; i < N; ++i){
            x[i] = lower_bound + (rand() / div); //NOTE: expensive?
        }
    }
    else{
        for(int i = 0; i < N; ++i){
            x[i] = initial_val;
        }
    }

    for(int i = 0; i < N; ++i) y[i] = 0.0;

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

void compare_with_mkl(
    const double *y,
    const double *mkl_result,
    const int N,
    bool verbose_compare
){
    double max_error = 0.0;
    double error;

    for (int i = 0; i < N; i++)
        max_error = std::max(max_error, y[i] - mkl_result[i]);
    printf("Max error: %f\n", max_error);

    if(verbose_compare){
        for (int i = 0; i < N; i++){
            printf("idx: %i, error: %f\n", i, y[i] - mkl_result[i]);
        }
    }
}

void validate_dp_result(
    CRSMtxData *crs_mat,
    double const *x,
    double const *y
){    
    int n_rows = crs_mat->n_rows;
    int n_cols = crs_mat->n_cols;

    double *mkl_result = new double[n_rows];

    char transa = 'n';
    double alpha = 1.0;
    double beta = 0.0; 
    char matdescra [4] = {
        'G', // general matrix
        ' ', // ignored
        ' ', // ignored
        'C'}; // zero-based indexing (C-style)

    // Computes y := alpha*A*x + beta*y, for A -> m * k, 
    // mkl_dcsrmv(transa, m, k, alpha, matdescra, val, indx, pntrb, pntre, x, beta, y)
    mkl_dcsrmv(
        &transa, 
        &n_rows, 
        &n_cols, 
        &alpha, 
        &matdescra[0], 
        crs_mat->val, 
        crs_mat->col, 
        crs_mat->row_ptr, 
        &(crs_mat->row_ptr)[1], 
        x, 
        &beta, 
        mkl_result
    );
    
    // Bool toggles verbose error checking
    compare_with_mkl(y, mkl_result, n_cols, false);
}