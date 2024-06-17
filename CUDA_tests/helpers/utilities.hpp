#ifndef UTILITIES_H
#define UTILITIES_H

#include "structs.hpp"

using ST=long;

// template <typename VT, typename IT>
void convert_to_scs(
    COOMtxData *local_mtx,
    ST C,
    ST sigma,
    SCSMtxData<double, int> *scs,
    int *work_sharing_arr = nullptr,
    int my_rank = 0
);

void apply_permutation(
    double *permuted_vec,
    double *vec_to_permute,
    int *perm,
    int num_elems_to_permute
);

void compare_with_mkl(
    const double *y,
    const double *mkl_result,
    const int N,
    bool verbose_compare
);

void compare_with_mkl(
    const double *y,
    const double *mkl_result,
    const int N,
    bool verbose_compare
);

void generate_x_and_y(
    double *x,
    double *y,
    int N,
    int nnz,
    bool rand_flag,
    double *values,
    double initial_val
);

void convert_to_crs(
    COOMtxData *coo_mat,
    CRSMtxData *crs_mat
);

void validate_dp_result(
    CRSMtxData *crs_mat,
    double const *x,
    double const *y
);

#endif