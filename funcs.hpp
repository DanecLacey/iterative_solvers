#ifndef FUNCS_H
#define FUNCS_H

#include <stdio.h>
#include <iostream>
#include <string>


#include "structs.hpp"
#include "kernels.hpp"

void assign_cli_inputs(
    int argc,
    char *argv[],
    std::string *matrix_file_name,
    std::string *solver_type
    );

void read_mtx(
    const std::string matrix_file_name,
    COOMtxData *coo_mat
    );

// TODO
//void convert_coo_to_crs(
//        );

void split_upper_lower_diagonal(
    COOMtxData *coo_mat,
    COOMtxData *U_coo_mat,
    COOMtxData *L_coo_mat,
    std::vector<double> *D_inv_coo_vec
);

void icr_vals(
    COOMtxData *coo_mat
);

inline void sort_perm(int *arr, int *perm, int len, bool rev=false);

void generate_vector(
    std::vector<double> *vec_to_populate,
    int size,
    bool rand_flag,
    int initial_val
);

double calc_residual(
    CRSMtxData *crs_mat,
    std::vector<double> *x_new,
    std::vector<double> *b
);

void summary_output(
    COOMtxData *coo_mat,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *residuals_vec,
    std::string *solver_type,
    LoopParams loop_params,
    Flags flags,
    double total_time_elapsed,
    double calc_time_elapsed
);

void iter_output(
    std::vector<double> *x_approx,
    int iter_count
);

void residuals_output(
    bool print_residuals,
    std::vector<double> *residuals_vec,
    int iter_count
);

void start_time(
    timeval *begin
);

double end_time(
    timeval *begin,
    timeval *end
);

void jacobi_iteration(
    CRSMtxData *crs_mat,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
);

void jacobi_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    CRSMtxData *crs_mat,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
);

void gs_iteration(
    CRSMtxData *crs_mat,
    std::vector<double> *b,
    std::vector<double> *x
);

void gs_solve(
    std::vector<double> *x,
    std::vector<double> *x_star,
    std::vector<double> *b,
    CRSMtxData *crs_mat,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
);

// void trivial_iteration(
//     COOMtxData *full_coo_L_plus_U_mtx,
//     std::vector<double> *D_inv_coo_vec,
//     std::vector<double> *neg_D_inv_coo_vec,
//     std::vector<double> *b,
//     std::vector<double> *x_old,
//     std::vector<double> *x_new
// );

// void trivial_solve(
//     std::vector<double> *x_old,
//     std::vector<double> *x_new,
//     std::vector<double> *x_star,
//     std::vector<double> *b,
//     COOMtxData *coo_mat,
//     std::vector<double> *residuals_vec,
//     double *calc_time_elapsed,
//     Flags *flags,
//     LoopParams *loop_params
// );

// void FOM_iteration(
//     COOMtxData *full_coo_L_plus_U_mtx,
//     std::vector<double> *D_inv_coo_vec,
//     std::vector<double> *neg_D_inv_coo_vec,
//     std::vector<double> *b,
//     std::vector<double> *x_old,
//     std::vector<double> *x_new
// );

// void FOM_solve(
//     std::vector<double> *x_old,
//     std::vector<double> *x_new,
//     std::vector<double> *x_star,
//     std::vector<double> *b,
//     COOMtxData *coo_mat,
//     std::vector<double> *residuals_vec,
//     double *calc_time_elapsed,
//     Flags *flags,
//     LoopParams *loop_params
// );

// void Arnoldi_iteration(
//     std::vector<double> *v_1,
//     COOMtxData *H_m,
//     COOMtxData *V_m,
//     int target_dim,
//     Flags *flags,
//     LoopParams *loop_params
// );



void write_residuals_to_file(std::vector<double> *residuals_vec);

double infty_vec_norm(
    const std::vector<double> *vec
);

double infty_mat_norm(
    CRSMtxData *crs_mat
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
#endif /*FUNCS_H*/
