#ifndef SOLVERS_H
#define SOLVERS_H

#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

void jacobi_iteration_ref(
    CRSMtxData *crs_mat,
    std::vector<double> *diag,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
);

void jacobi_iteration_sep_diag_subtract(
    CRSMtxData *crs_mat,
    std::vector<double> *diag,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
);

void jacobi_iteration_sep(
    CRSMtxData *crs_mat,
    std::vector<double> *diag,
    std::vector<double> *b,
    std::vector<double> *x_old,
    std::vector<double> *x_new
);

void jacobi_solve(
    std::vector<double> *x_old,
    std::vector<double> *x_new,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *A_x_tmp,
    CRSMtxData *crs_mat,
    std::vector<double> *diag,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
);

void gs_iteration_ref(
    CRSMtxData *crs_mat,
    CRSMtxData *crs_L,
    CRSMtxData *crs_U,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
);

void gs_iteration_sep(
    CRSMtxData *crs_mat,
    CRSMtxData *crs_L,
    CRSMtxData *crs_U,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *b,
    std::vector<double> *x
);

void gs_solve(
    std::vector<double> *x,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *r,
    std::vector<double> *tmp,
    CRSMtxData *crs_mat,
    CRSMtxData *crs_L,
    CRSMtxData *crs_U,
    std::vector<double> *diag,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
);
#endif /*SOLVERS_H*/