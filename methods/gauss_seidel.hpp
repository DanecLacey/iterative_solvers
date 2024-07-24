#ifndef GAUSS_SEIDEL_H
#define GAUSS_SEIDEL_H

#include "../sparse_matrix.hpp"

void gs_iteration_ref_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x
);

void gs_iteration_sep_cpu(
    SparseMtxFormat *sparse_mat,
    double *tmp,
    double *D,
    double *b,
    double *x,
    int N
);

void init_gs_structs(
    COOMtxData *coo_mat,
    SparseMtxFormat *sparse_mat
);

#endif