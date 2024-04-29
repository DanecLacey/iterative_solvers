#ifndef KERNELS_H
#define KERNELS_H
#include "structs.hpp"
#include <cstdlib>


void mtx_spmv_coo(
    std::vector<double> *y,
    const COOMtxData *mtx,
    const std::vector<double> *x
    );

void vec_spmv_coo(
    std::vector<double> *mult_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
    );

void sum_vectors(
    std::vector<double> *sum_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
);

double infty_vec_norm_cpu(
    const std::vector<double> *vec
);

#ifdef __CUDACC__
__global__ 
void infty_vec_norm_gpu(
    const double *d_vec,
    double *d_infty_norm,
    double n_rows
);
#endif

void subtract_vectors_cpu(
    std::vector<double> *result_vec,
    const std::vector<double> *vec1,
    const std::vector<double> *vec2
);

// TODO: Do we also need to define __global__ in the header
#ifdef __CUDACC__
void subtract_vectors_gpu(
    double *result_vec,
    double *vec1,
    double *vec2
);
#endif

void sum_matrices(
    COOMtxData *sum_mtx,
    const COOMtxData *mtx1,
    const COOMtxData *mtx2
);

void spmv_crs_cpu(
    std::vector<double> *y,
    const CRSMtxData *crs_mat,
    const std::vector<double> *x
);

#ifdef __CUDACC__
void spmv_crs_gpu(
    const double *val,
    const int *col,
    const int *row_ptr,
    double *y,
    const double *x,
    const int n_rows
);
#endif

void jacobi_normalize_x_cpu(
    std::vector<double> *x_new,
    const std::vector<double> *x_old,
    const std::vector<double> *D,
    const std::vector<double> *b,
    int n_rows
);

#ifdef __CUDACC__
__global__
void jacobi_normalize_x_gpu(
    double *d_x_new,
    const double *d_x_old,
    const double *d_D,
    const double *d_b,
    int d_n_rows
);
#endif

void spltsv_crs(
    const CRSMtxData *crs_L,
    std::vector<double> *x,
    const std::vector<double> *D,
    const std::vector<double> *b_Ux
);

void calc_residual_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *x,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *tmp
);

#ifdef __CUDACC__
__global__
void calc_residual_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_x,
    double *d_r,
    double *d_b,
    double *d_tmp,
    int d_n_rows
);
#endif

#endif /*KERNELS_H*/