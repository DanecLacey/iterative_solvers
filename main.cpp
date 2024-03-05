#include <vector>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <sys/time.h>

// #include "structs.hpp"
#include "funcs.hpp"
#include "mmio.h"

void solve(
    CRSMtxData *crs_mat,
    std::string *solver_type,
    std::vector<double> *x_star,
    std::vector<double> *b,
    std::vector<double> *residuals_vec,
    double *calc_time_elapsed,
    Flags *flags,
    LoopParams *loop_params
){
    int n_cols = crs_mat->n_cols;

    // Declare common structs
    std::vector<double> x_new(n_cols);
    std::vector<double> x_old(n_cols);

    // Pre-allocate vector sizes
    x_star->resize(n_cols);
    b->resize(n_cols);

    // TODO: What are the ramifications of having x and b different scales than the data?
    // And how to make the "same scale" as data?
    // Make b vector
    // NOTE: Currently using random vector b with elements \in (0, 10)
    generate_vector(b, n_cols, false, 1);
    // ^ b should likely draw from A(min) to A(max) range of values

    // Make initial x vector
    // NOTE: Currently taking the 1 vector as starting guess
    generate_vector(&x_old, n_cols, false, 2);

    // Precalculate stopping criteria
    double infty_norm_A = infty_mat_norm(crs_mat);
    double infty_norm_b = infty_vec_norm(b);
    loop_params->stopping_criteria = loop_params->tol * calc_residual(crs_mat, &x_old, b);

    if ((*solver_type) == "jacobi"){
        jacobi_solve(&x_old, &x_new, x_star, b, crs_mat, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    else if ((*solver_type) == "gauss-seidel"){
        gs_solve(&x_old, x_star, b, crs_mat, residuals_vec, calc_time_elapsed, flags, loop_params);
    }
    // else if ((*solver_type) == "trivial"){
    //     trivial_solve(&x_old, &x_new, x_star, b, full_crs_mat, residuals_vec, calc_time_elapsed, flags, loop_params);
    // }
    // else if ((*solver_type) == "FOM"){
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
        // TODO
        // FOM_solve(&x_old, &x_new, x_star, b, full_coo_mtx, residuals_vec, calc_time_elapsed, flags, loop_params);
    // }
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
        false, // print_iters
        true, // print_summary
        true, // print_residuals
        true, // convergence_flag. TODO: really shouldn't be here
        true // export simple error per iteration in CURRDIR/errors.txt
    };

    LoopParams loop_params{
        0, // init iteration count
        0, // init residuals count
        5, // calculate residual every n iterations
        500, // maximum iteration count
        0.0, // init stopping criteria
        1e-15// tolerance to stop iterations
    };

    COOMtxData *coo_mat = new COOMtxData;
    CRSMtxData *crs_mat = new CRSMtxData;

    std::vector<double> x_star, b;
    std::vector<double> residuals_vec(loop_params.max_iters / loop_params.residual_check_len + 1);
    double total_time_elapsed;
    double calc_time_elapsed;

    // Store mtx file name and type of solver
    assign_cli_inputs(argc, argv, &matrix_file_name, &solver_type);

    // Read COO format mtx file
    read_mtx(matrix_file_name, coo_mat);

    // TODO
    // preprocessing plus timers
    coo_mat->convert_to_crs(crs_mat);

    // Solve Ax=b given a random b vector, and time it
    start_time(&total_time_start);

    solve(crs_mat, &solver_type, &x_star, &b, &residuals_vec, &calc_time_elapsed, &flags, &loop_params);

    total_time_elapsed = end_time(&total_time_start, &total_time_end);

    if(flags.print_summary){
        summary_output(coo_mat, &x_star, &b, &residuals_vec, &solver_type, loop_params, flags, total_time_elapsed, calc_time_elapsed);
    }
    if(flags.export_errors){
        write_residuals_to_file(&residuals_vec);
    }

    delete coo_mat;
    delete crs_mat;

    return 0;
}