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
    int initial_val
);

double calc_residual(
    const CRSMtxData *crs_mat,
    const std::vector<double> *x_new,
    const std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *A_x_tmp
);

void start_time(
    timeval *begin
);

double end_time(
    timeval *begin,
    timeval *end
);

double infty_vec_norm(
    const std::vector<double> *vec
);

double infty_mat_norm(
    const CRSMtxData *crs_mat
);

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
#endif /*UTILITY_FUNCS_H*/
