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

void sum_matrices(
    COOMtxData *sum_mtx,
    const COOMtxData *mtx1,
    const COOMtxData *mtx2
);

void spmv_crs(
    std::vector<double> *y,
    const CRSMtxData *crs_mat,
    const std::vector<double> *x
    );

#endif /*KERNELS_H*/