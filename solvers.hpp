#ifndef SOLVERS_H
#define SOLVERS_H

#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"

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
#endif /*SOLVERS_H*/