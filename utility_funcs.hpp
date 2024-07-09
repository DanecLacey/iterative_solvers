#ifndef UTILITY_FUNCS_H
#define UTILITY_FUNCS_H

#include <stdio.h>
#include <iostream>
#include <string>

#include "structs.hpp"
#include "kernels.hpp"


void generate_vector(
    std::vector<double> *vec_to_populate,
    int size,
    bool rand_flag,
    double initial_val
);

void start_time(
    timeval *begin
);

double end_time(
    timeval *begin,
    timeval *end
);


// Do you need this anywhere??
// double infty_mat_norm(
//     const CRSMtxData *crs_mat
// );

void gen_neg_inv(
    std::vector<double> *neg_inv_coo_vec,
    std::vector<double> *inv_coo_vec,
    std::vector<double> *coo_vec
);

void recip_elems(    
    std::vector<double> *recip_vec,
    std::vector<double> *vec
);

void extract_diag(
    const CRSMtxData *crs_mat,
    std::vector<double> *diag
);

void compare_with_direct(
    CRSMtxData *crs_mat,
    std::string matrix_file_name,
    LoopParams loop_params,
    std::vector<double> *x_star,
    double iterative_final_residual
);

void split_upper_lower_diagonal(
    COOMtxData *full_coo_mtx,
    COOMtxData *U_coo_mtx,
    COOMtxData *L_coo_mtx,
    std::vector<double> *D_coo_vec
);

void convert_to_crs(
    COOMtxData *coo_mat,
    CRSMtxData *crs_mat
);

void gmres_get_x(
    double *R,
    double *g,
    double *x,
    double *x_0,
    double *V,
    int n_rows,
    int iter_count,
    int max_gmres_iters
);

void preprocessing(
    argType *args
);

void init_gmres_structs(
    argType *args,
    int n_rows
);

void record_residual_norm(
    argType *args,
    Flags *flags,
    SparseMtxFormat *sparse_mat,
    double *residual_norm,
    double *r,
    double *x,
    double *b,
    double *x_new,
    double *tmp
);

void iter_output(
    const double *x_approx,
    int N,
    int iter_count
);

void print_x(
    argType *args,
    double *x,
    double *x_new,
    double *x_old,
    int n_rows
);

void init(
    double *vec,
    double val,
    int size
);
#endif /*UTILITY_FUNCS_H*/
