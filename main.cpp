#include <vector>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <sys/time.h>

#ifdef USE_EIGEN
#include <Eigen/SparseLU>
#include <unsupported/Eigen/SparseExtra>
#endif

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif

#include "utility_funcs.hpp"
#include "io_funcs.hpp"
#include "solvers.hpp"
#include "structs.hpp"
#include "mmio.h"

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
     *      Conjugate Gradient
     *      GMRES(restart length)
     * until relative tolerance reached
     *      "||b - Ax_k|| / ||b - Ax_0|| < tol"
     * is reached
     * 4. Optionally:
     *      Report pre-processing and calculation times
     *      Report number of iterations and validate A * x_star = b
     *      Export errors per iteration to external text file
     * */

    bogus_init_pin();

#ifdef USE_LIKWID
    LIKWID_MARKER_INIT;
#endif

    // Declare and init input structs
    argType *args = new argType;
    Timers *timers = new Timers;
    args->timers = timers;
    
    timeval *total_time_start = new timeval;
    timeval *total_time_end = new timeval;
    Stopwatch *total_wtime = new Stopwatch(total_time_start, total_time_end);
    args->timers->total_wtime = total_wtime;
    args->timers->total_wtime->start_stopwatch();

    std::string matrix_file_name;
    assign_cli_inputs(args, argc, argv, &matrix_file_name);

    Solver *solver = new Solver(args->solver_type);
    Preconditioner *preconditioner = new Preconditioner;
    gmresArgs *gmres_args = new gmresArgs;
    args->solver = solver;
    args->preconditioner = preconditioner;
    args->solver->gmres_args = gmres_args;

    SparseMtxFormat *sparse_mat = new SparseMtxFormat;
    COOMtxData *coo_mat = new COOMtxData;

#ifdef USE_SCAMAC
    std::cout << "Generating Matrix" << std::endl;
    matrix_file_name = args->scamac_args;
    scamac_make_mtx(args, coo_mat);
#else
    std::cout << "Reading Matrix" << std::endl;
    read_mtx(matrix_file_name, coo_mat);
#endif

//////////////// User Parameters ////////////////

    Flags flags{
        false, // print_iters. WARNING: costly
        true, // print_summary
        true, // print_residuals
        false, // convergence_flag. TODO: really shouldn't be here
        false, // apply preconditioner TODO: not implemented
        false, // Compare to SparseLU direct solver
        false // generate random data for b and initial x vectors
    };

    // TODO: Split struct into const and nonconst fields
    LoopParams loop_params{
        0, // init iteration count
        0, // init residuals count
        1, // calculate residual every n iterations
        MAX_ITERS, // maximum iteration count
        0.0, // init stopping criteria
        TOL, // tolerance to stop iterations
        1.0, // init value for b
        0.0, // init value for x
        GMRES_RESTART_LEN // GMRES restart length
    };
////////////////////////////////////////////////

    args->loop_params = &loop_params;
    args->flags = &flags;
    args->matrix_file_name = &matrix_file_name;
    args->sparse_mat = sparse_mat;
    args->solver->gmres_args->restart_length = loop_params.gmres_restart_len;
    args->coo_mat = coo_mat;

#ifdef __CUDACC__
    double *d_x_star = new double;
    double *d_x_new = new double;
    double *d_x_old = new double;
    int *d_row_ptr = new int;
    int *d_col = new int;
    double *d_val = new double;
    double *d_tmp = new double;
    double *d_r = new double;
    double *d_D = new double;
    double *d_b = new double;
    double *d_normed_residuals = new double;

    args->solver->d_x_star = d_x_star;
    args->solver->d_x_new = d_x_new;
    args->solver->d_x_old = d_x_old;
    args->solver->d_row_ptr = d_row_ptr;
    args->solver->d_col = d_col;
    args->solver->d_val = d_val;
    args->solver->d_tmp = d_tmp;
    args->solver->d_r = d_r;
    args->solver->d_D = d_D;
    args->solver->d_b = d_b;
    args->d_normed_residuals = d_normed_residuals;
#endif

#ifdef USE_USPMV
    ScsData<double, int> *scs_mat = new ScsData<double, int>;
    args->sparse_mat->scs_mat = scs_mat;
#ifdef USE_AP
    ScsData<double, int> *scs_mat_hp = new ScsData<double, int>;
    ScsData<float, int> *scs_mat_lp = new ScsData<float, int>;

    args->sparse_mat->scs_mat_hp = scs_mat_hp;
    args->sparse_mat->scs_mat_lp = scs_mat_lp;
#endif
    ScsData<double, int> *scs_L = new ScsData<double, int>;
    ScsData<double, int> *scs_U = new ScsData<double, int>;
    
    args->sparse_mat->scs_L = scs_L;
    args->sparse_mat->scs_U = scs_U;
#endif

    CRSMtxData *crs_mat = new CRSMtxData;
    CRSMtxData *crs_L = new CRSMtxData;
    CRSMtxData *crs_U = new CRSMtxData;
    args->sparse_mat->crs_mat = crs_mat;
    args->sparse_mat->crs_L = crs_L;
    args->sparse_mat->crs_U = crs_U;

    preprocessing(args);

    solve(args);

    postprocessing(args);

#ifdef USE_USPMV
    delete scs_mat;
    delete scs_L;
    delete scs_U;
#else
    delete crs_mat;
    delete crs_L;
    delete crs_U;
#endif

    delete coo_mat;
    delete sparse_mat;
    delete args;
    delete solver;
    delete preconditioner;

#ifdef __CUDACC__
    cudaFree(args->solver->d_x_star);
    cudaFree(args->solver->d_x_new);
    cudaFree(args->solver->d_x_old);
    cudaFree(args->solver->d_row_ptr);
    cudaFree(args->solver->d_col);
    cudaFree(args->solver->d_val);
    cudaFree(args->solver->d_tmp);
    cudaFree(args->solver->d_r);
    cudaFree(args->solver->d_D);
    cudaFree(args->solver->d_b);
    cudaFree(args->d_normed_residuals);
#endif

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    return 0;
}
