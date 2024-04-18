#include <vector>
#include <string>
#include <stdbool.h>
#include <algorithm>
#include <sys/time.h>

#ifdef USE_EIGEN
    #include <Eigen/SparseLU>
    #include <unsupported/Eigen/SparseExtra>
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
     *      Conjugate Gradient (under development)
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
        false, // Compare to SparseLU direct solver
        false // generate random data for b and initial x vectors
    };

    // TODO: Split struct into const and nonconst fields
    LoopParams loop_params{
        0, // init iteration count
        0, // init residuals count
        1, // calculate residual every n iterations
        3000, // maximum iteration count
        0.0, // init stopping criteria
        1e-14, // tolerance to stop iterations
        1.1, // init value for b
        3.0 // init value for x
    };

    argType *args = new argType;
    SparseMtxFormat *sparse_mat = new SparseMtxFormat;
    std::vector<double> x_star; 
    std::vector<double> x_new;
    std::vector<double> x_old;
    std::vector<double> tmp;
    std::vector<double> D;
    std::vector<double> r; 
    std::vector<double> b;
    std::vector<double> normed_residuals(loop_params.max_iters / loop_params.residual_check_len + 1);
    
    args->coo_mat = coo_mat;
    args->x_star = &x_star;
    args->x_new = &x_new;
    args->x_old = &x_old;
    args->tmp = &tmp;
    args->D = &D;
    args->r = &r;
    args->b = &b;
    args->normed_residuals = &normed_residuals;
    args->loop_params = &loop_params;
    args->solver_type = solver_type;
    args->flags = &flags;
    args->matrix_file_name = &matrix_file_name;
    args->sparse_mat = sparse_mat;

#ifdef USE_USPMV
    ScsData<double, int> *scs_mat = new ScsData<double, int>;
    ScsData<double, int> *scs_L = new ScsData<double, int>;
    ScsData<double, int> *scs_U = new ScsData<double, int>;
    args->sparse_mat->scs_mat = scs_mat;
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

    args->total_time_elapsed = end_time(&total_time_start, &total_time_end);

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

    return 0;
}