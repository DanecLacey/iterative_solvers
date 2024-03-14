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
    std::string solver_type,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *normed_residuals,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){
    int n_cols = crs_mat->n_cols;

    // Declare common structs and allocate sizes
    std::vector<double> x_new(n_cols, 0);
    std::vector<double> x_old(n_cols, 0);
    std::vector<double> r(n_cols, 0);
    std::vector<double> diag(n_cols, 0);
    std::vector<double> A_x_tmp(n_cols, 0); // temporary buffer for residual computations
    x_star->resize(n_cols, 0);
    b->resize(n_cols, 0);

    // TODO: What are the ramifications of having x and b different scales than the data? And how to make the "same scale" as data?
    // Make b vector
    generate_vector(b, n_cols, false, loop_params->init_b);
    // ^ b should likely draw from A(min) to A(max) range of values

    // Make initial x vector
    generate_vector(&x_old, n_cols, false, loop_params->init_x);

    // Extract diagonal elements for easier library interop with kernels
    extract_diag(crs_mat, &diag);

    // Precalculate stopping criteria
    calc_residual(crs_mat, &x_old, b, &r, &A_x_tmp);
    loop_params->stopping_criteria = loop_params->tol * infty_vec_norm(&r); 

    if (solver_type == "jacobi"){
        jacobi_solve(&x_old, &x_new, x_star, b, &r, &A_x_tmp, crs_mat, &diag, normed_residuals, calc_time_elapsed, flags, loop_params);
    }
    else if (solver_type == "gauss-seidel"){
        gs_solve(&x_old, x_star, b, &r, &A_x_tmp, crs_mat, &diag, normed_residuals, calc_time_elapsed, flags, loop_params);
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
     *          x_k+1 = D^{-1} * (b - (L + U) * x_k)
     *      Gauss-Seidel Iteration
     *          x_k+1 = (D+L)^{-1} * (b - R * x_k) 
     * until tolerance
     *      "||b - A * x_k|| / ||b - A * x_0|| < tol"
     * is reached
     * 4. Optionally:
     *      Report pre-processing and calculation times
     *      Report number of iterations and validate A * x_star = b
     *      Export errors per iteration to external text file
     * */

    // Declare structs
    std::string matrix_file_name, solver_type;
    struct timeval total_time_start, total_time_end;
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
        10000, // maximum iteration count
        0.0, // init stopping criteria
        1e-14, // tolerance to stop iterations
        1.1, // init value for b
        0.0 // init value for x
    };

    COOMtxData *coo_mat = new COOMtxData;
    CRSMtxData *crs_mat = new CRSMtxData;

    std::vector<double> x_star, b;
    std::vector<double> normed_residuals(loop_params.max_iters / loop_params.residual_check_len + 1);
    double total_time_elapsed;
    double calc_time_elapsed;

    // Store mtx file name and type of solver
    assign_cli_inputs(argc, argv, &matrix_file_name, &solver_type);

    // Read COO format mtx file
    read_mtx(matrix_file_name, coo_mat);

    // Collect preprocessing time
    start_time(&total_time_start);

    // TODO: MPI preprocessing will go here
         
    coo_mat->convert_to_crs(crs_mat);

    // TODO: process-local preprocessing
    // preprocessing();

    ///////// Main solver routine /////////
    solve(crs_mat, solver_type, &x_star, &b, &normed_residuals, &calc_time_elapsed, &flags, &loop_params);

    total_time_elapsed = end_time(&total_time_start, &total_time_end);

    postprocessing(
        crs_mat, 
        matrix_file_name, 
        loop_params, 
        &x_star, 
        &b,
        &normed_residuals,
        solver_type, 
        flags, 
        total_time_elapsed, 
        calc_time_elapsed
    );

    delete crs_mat;
    delete coo_mat;

    return 0;
}