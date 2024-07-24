#include "../structs.hpp"

void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new // treat like y
);

void jacobi_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new,
    int N
);

#ifdef __CUDACC__
void jacobi_iteration_ref_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new,
    int n_rows
);

void jacobi_iteration_sep_gpu(
    int d_n_rows,
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new
);
#endif