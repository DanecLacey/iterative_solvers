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
    const double *vec,
    int N
);

void dense_MMM(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
);

void dense_MMM_t(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
);

void dense_MMM_t_t(
    double *A,
    double *B,
    double *C,
    int n_rows_A,
    int n_cols_A,
    int n_cols_B
);

void scale(
    double *result_vec,
    const double *vec,
    const double scalar,
    int N
);

double euclidean_vec_norm_cpu(
    const double *vec,
    int N
);

void dense_transpose(
    const double *mat,
    double *mat_t,
    int n_rows,
    int n_cols
);

void dot(
    const double *vec1,
    const double *vec2,
    double *result,
    int N
);

void dot_od(
    const double *vec1,
    const double *vec2,
    double *partial_sum,
    int N
);

void strided_1_dot(
    const double *vec1,
    const double *vec2,
    double *result,
    int N,
    int stride
);


void strided_2_dot(
    const double *vec1,
    const double *vec2,
    double *result,
    int N,
    int stride
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
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N,
    double scale = 1.0
);

void subtract_vectors_cpu_od(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N,
    double scale = 1.0
);

#ifdef __CUDACC__
__global__
void subtract_vectors_gpu(
    double *result_vec,
    const double *vec1,
    const double *vec2,
    int N
);
#endif

void sum_matrices(
    COOMtxData *sum_mtx,
    const COOMtxData *mtx1,
    const COOMtxData *mtx2
);

void spmv_crs_cpu(
    double *y,
    const CRSMtxData *crs_mat,
    double *x
);

#ifdef __CUDACC__
__global__
void spmv_crs_gpu(
    const int d_n_rows,
    const int *d_row_ptr,
    const int *d_col,
    const double *d_val,
    const double *d_x,
    double *d_y
    );
#endif

void jacobi_normalize_x_cpu(
    double *x_new,
    const double *x_old,
    const double *D,
    const double *b,
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
    double *x,
    const double *D,
    const double *b_Ux
);

void calc_residual_cpu(
    SparseMtxFormat *sparse_mat,
    double *x,
    double *b,
    double *r,
    double *tmp,
    int N
);

#ifdef __CUDACC__
void calc_residual_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_x,
    double *d_b,
    double *d_r,
    double *d_tmp,
    int d_n_rows
);
#endif

#endif /*KERNELS_H*/