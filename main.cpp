#include <vector>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <sys/time.h>

#ifdef USE_EIGEN
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#endif

#include "utility_funcs.hpp"
#include "io_funcs.hpp"
#include "solvers.hpp"
#include "mmio.h"

void solve(
    CRSMtxData *crs_mat,
    CRSMtxData *crs_L,
    CRSMtxData *crs_U,
    std::vector<double> *x_star,
    std::vector<double> *x_new,
    std::vector<double> *x_old,
    std::vector<double> *b,
    std::vector<double> *tmp,
    std::vector<double> *D,
    std::vector<double> *r,
    std::vector<double> *normed_residuals,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params,
    std::string solver_type
){

    if (solver_type == "jacobi"){
        jacobi_solve(x_old, x_new, x_star, b, r, tmp, crs_mat, D, normed_residuals, calc_time_elapsed, flags, loop_params);
    }
    else if (solver_type == "gauss-seidel"){
        gs_solve(x_old, x_star, b, r, tmp, crs_mat, crs_L, crs_U, D, normed_residuals, calc_time_elapsed, flags, loop_params);
    }
    else{
        printf("ERROR: solve: This solver method is not implemented yet.\n");
        exit(1);
    }
}

int main(int argc, char *argv[]){
    /* Basic structure:
     * 1. Form a matrix A (Just read .mtx with mmio and convert to CRS)
            Optionally perform preprocessing steps on A
     * 2. "Randomly" form b vector and initial x vector
     * 3. Perform solver steps:
     *      Jacobi Iteration:
     *          x_k = D^{-1}(b - (L + U)x_{k-1})
     *      Gauss-Seidel Iteration
     *          x_k = (D+L)^{-1}(b - Ux_{k-1}) 
     * until tolerance
     *      "||b - Ax_k|| / ||b - Ax_0|| < tol"
     * is reached
     * 4. Optionally:
     *      Report pre-processing and calculation times
     *      Report number of iterations and validate A * x_star = b
     *      Export errors per iteration to external text file
     * */

    struct timeval total_time_start, total_time_end;
    start_time(&total_time_start);

    // Declare and init input structs
    std::string matrix_file_name, solver_type;
    assign_cli_inputs(argc, argv, &matrix_file_name, &solver_type);

    COOMtxData *coo_mat = new COOMtxData;
    read_mtx(matrix_file_name, coo_mat);

    Flags flags{
        false, // print_iters. WARNING: costly
        true, // print_summary
        true, // print_residuals
        true, // convergence_flag. TODO: really shouldn't be here
        false, // apply preconditioner TODO: not implemented
        false // Compare to SparseLU direct solver
    };

    LoopParams loop_params{
        0, // init iteration count
        0, // init residuals count
        50, // calculate residual every n iterations
        3000, // maximum iteration count
        0.0, // init stopping criteria
        1e-14, // tolerance to stop iterations
        1.1, // init value for b
        0.0 // init value for x
    };

    // Declare and init common structs
    CRSMtxData *crs_mat = new CRSMtxData;
    CRSMtxData *crs_L = new CRSMtxData;
    CRSMtxData *crs_U = new CRSMtxData;
    std::vector<double> x_star, b;
    std::vector<double> normed_residuals(loop_params.max_iters / loop_params.residual_check_len + 1);
    double total_time_elapsed;
    double calc_time_elapsed;
    std::vector<double> x_new(coo_mat->n_cols, 0);
    std::vector<double> x_old(coo_mat->n_cols, 0);
    std::vector<double> r(coo_mat->n_cols, 0);
    std::vector<double> D(coo_mat->n_cols, 0);
    std::vector<double> tmp(coo_mat->n_cols, 0); // temporary buffer for residual computations

    preprocessing(
        coo_mat,
        crs_mat,
        crs_L,
        crs_U,
        &x_star,
        &x_new,
        &x_old,
        &tmp,
        &D,
        &r,
        &b,
        &loop_params,
        solver_type
    );

    solve(
        crs_mat,
        crs_L,
        crs_U,
        &x_star,
        &x_new,
        &x_old,
        &b,
        &tmp,
        &D,
        &r,
        &normed_residuals,
        &calc_time_elapsed,
        &flags,
        &loop_params,
        solver_type
    );

    total_time_elapsed = end_time(&total_time_start, &total_time_end);

    postprocessing(
        crs_mat, 
        matrix_file_name, 
        &x_star, 
        &b,
        &normed_residuals,
        flags, 
        total_time_elapsed, 
        calc_time_elapsed,
        loop_params,
        solver_type
    );

    delete crs_mat;
    delete coo_mat;

    return 0;
}