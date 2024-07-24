#ifndef SOLVERS_H
#define SOLVERS_H

#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"
#include "methods/jacobi.hpp"


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
    argType *args
);
#endif

void solve(
    argType *args
);
#endif /*SOLVERS_H*/