#include <vector>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <sys/time.h>

#include "structs.hpp"
#include "funcs.hpp"
#include "mmio.h"

void solve(
    COOMtxData *full_coo_mtx,
    std::string *solver_type,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){
    int ncols = full_coo_mtx->n_cols;

    // Declare common structs
    std::vector<double> x_new(ncols);
    std::vector<double> x_old(ncols);

    // Pre-allocate vector sizes
    x_star->resize(ncols);
    b->resize(ncols);


    // TODO: What are the ramifications of having x and b different scales than the data?
    // And how to make the "same scale" as data?
    // Make b vector
    // NOTE: Currently using random vector b with elements \in (0, 10)
    // generate_vector(b, ncols, true, NULL);
    generate_vector(b, ncols, false, 1);
    // ^ b should likely draw from A(min) to A(max) range of values

    // Make initial x vector
    // NOTE: Currently taking the 1 vector as starting guess
    generate_vector(&x_old, ncols, false, 2);

    if ((*solver_type) == "jacobi"){
        jacobi_solve(&x_old, &x_new, x_star, b, full_coo_mtx, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    else if ((*solver_type) == "gauss-seidel"){
        gs_solve(&x_old, &x_new, x_star, b, full_coo_mtx, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    else if ((*solver_type) == "trivial"){
        trivial_solve(&x_old, &x_new, x_star, b, full_coo_mtx, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    else if ((*solver_type) == "FOM"){
    // FOM
    // 0. Choose max subspace dim m
    // 1. Initialize r_0 = b - A * x_0, beta = || r_0 ||_2, v_1 = r_0/beta, and H_m = 0
    // 2. For j = 1,2,...,m
    // 3.   Compute w_j = A * v_j
    // 4.   For i = 1,...,j
    // 5.       h_ij = v_i^T * w_j
    // 6.       w_j = w_j - h_ij * v_i
    // 7.   Compute h_{j+1},j = || w_j ||_2 (if h_{j+1},j = 0, set m = j, goto 9.)
    // 8.   Compute v_{j+1} = w_j/h_{j+1},j
    // 8.1  Place into V_m[j+1]
    // 9. Compute y_m = H_m^{-1}(beta * e_1), x_m = x_0 + V_m * y_m
        FOM_solve(&x_old, &x_new, x_star, b, full_coo_mtx, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    else{
        printf("ERROR: solve: This solver method is not implemented yet.\n");
        exit(1);
    }
}

// TODO: link with some library to compute eigenvalues. Having a spectral radius less than one
// would guarentee convergence, and help spot when there are problems in the code.
int main(int argc, char *argv[]){
    /* Basic structure:
     * 1. Form a matrix A (Just read .mtx with mmio)
     * 2. "Randomly" form b vector and initial x vector
     * 3. Depending on the solver, seperate A = D + L + R
     * 4. Perform solver steps:
     *      Jacobi Iteration:
     *          x_k+1 = D^{-1} * (b - (L + U) * x_k)
     *      Gauss-Seidel Iteration
     *          x_k+1 = (D+L)^{-1} * (b - R * x_k) 
     *      Trivial Iteration
     *          x_k+1 = (I - A) * x_k + b
     * until (Dongarra) tolerance 1.
     *      "||A * x_k - b|| < tol * (||A|| * ||x_k|| + ||b||)"
     * or (Ketchup) tolerance 2.
     *      "|| (x_new - x_old) / x_old || < tol"
     * is reached
     * 5. Optionally:
     *      Report pre-processing and calculation times
     *      Report number of iterations and validate A * x_star = b
     *      Export errors per iteration to external text file
     * */

    // Declare structs
    std::string matrix_file_name, solver_type;
    COOMtxData full_coo_mtx;
    struct timeval total_time_start, total_time_end;
    Flags flags{
        false, // print_iters
        true, // print_summary
        true, // print_residuals
        true, // convergence_flag. TODO: really shouldn't be here
        true // export simple error per iteration in CURRDIR/errors.txt
    };
    LoopParams loop_params{
        0, // current iteration count
        500, // maximum iteration count
        1e-6// tolerance to stop iterations
    };

    // TODO:
    // CRSMtxData U_crs_mtx, L_crs_mtx, D_crs_mtx, D_inv_crs_mtx;

    std::vector<double> x_star, b;
    std::vector<double> residuals_vec(loop_params.max_iters);
    double total_time_elapsed;
    double calc_time_elapsed;

    // Store mtx file name and type of solver
    assign_cli_inputs(argc, argv, &matrix_file_name, &solver_type);

    // Read COO format mtx file
    read_mtx(matrix_file_name, &full_coo_mtx);

    // Solve Ax=b given a random b vector, and time it
    start_time(&total_time_start);
    solve(&full_coo_mtx, &solver_type, &x_star, &b, &residuals_vec, &calc_time_elapsed, &flags, &loop_params);
    total_time_elapsed = end_time(&total_time_start, &total_time_end);

    if(flags.print_summary){
        summary_output(&full_coo_mtx, &x_star, &b, &residuals_vec, &solver_type, loop_params.max_iters, flags.convergence_flag, flags.print_residuals, loop_params.iter_count, total_time_elapsed, calc_time_elapsed, loop_params.tol);
    }
    if(flags.export_errors){
        write_residuals_to_file(&residuals_vec);
    }

    return 0;
}