#ifndef SOLVERS_H
#define SOLVERS_H

#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

void jacobi_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new 
);

void jacobi_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new 
);

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
);

void gs_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
);

void solve_cpu(
    argType *args
);

#ifdef __CUDACC__
void solve_gpu(
    argType *args
);
#endif

void solve(
    argType *args
);
#endif /*SOLVERS_H*/