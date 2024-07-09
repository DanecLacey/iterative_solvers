#ifndef SOLVERS_H
#define SOLVERS_H

#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new
);

void jacobi_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *D,
    double *b,
    double *x_old,
    double *x_new,
    int N
);

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
);

void gs_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
);

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
);

void gm_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *V,
    double *H,
    double *H_tmp,
    double *J,
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
);

void solve_cpu(
    argType *args,
    LoopParams *loop_params
);

#ifdef __CUDACC__
__global__
void jacobi_iteration_ref_gpu(
    int *d_row_ptr,
    int *d_col,
    double *d_val,
    double *d_D,
    double *d_b,
    double *d_x_old,
    double *d_x_new
);

// NOTE: Doesn't need a qualifier, since only a launcher
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

void solve_gpu(
    argType *args,
    LoopParams *loop_params
);
#endif

void solve(
    argType *args,
    LoopParams *loop_params
);
#endif /*SOLVERS_H*/