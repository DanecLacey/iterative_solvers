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
     *      GMRES (under development)
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
    argType *args = new argType;
    std::string matrix_file_name;
    assign_cli_inputs(args, argc, argv, &matrix_file_name);

    COOMtxData *coo_mat = new COOMtxData;
    read_mtx(matrix_file_name, coo_mat);

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
        50000, // maximum iteration count
        0.0, // init stopping criteria
        1e-13, // tolerance to stop iterations
        21.1, // init value for b
        3.0, // init value for x
        20 // GMRES restart length
    };

    
    SparseMtxFormat *sparse_mat = new SparseMtxFormat;

    args->vec_size = coo_mat->n_cols;
    double *x_star = new double[args->vec_size];
    double *x_old = new double[args->vec_size];
    double *x_new = new double[args->vec_size];
    double *tmp = new double[args->vec_size];
    double *D = new double[args->vec_size];
    double *r = new double[args->vec_size];
    double *b = new double[args->vec_size];
    double *normed_residuals = new double[loop_params.max_iters / loop_params.residual_check_len + 1];

    double *init_v;
    double *V;
    double *H;
    double *H_tmp;
    double *J;
    double *R;
    double *Q;
    double *Q_copy;
    double *g;
    double *g_copy; 

    if(args->solver_type == "gmres"){
        std::cout << "Allocating space" << std::endl;
        double *init_v = new double[args->vec_size];
        double *V = new double[sparse_mat->crs_mat->n_rows * loop_params.gmres_restart_len]; // (m x n)
        double *H = new double[(loop_params.gmres_restart_len+1) * loop_params.gmres_restart_len]; // (m+1 x m) 
        double *H_tmp = new double[(loop_params.gmres_restart_len+1) * loop_params.gmres_restart_len]; // (m+1 x m)
        double *J = new double[(loop_params.gmres_restart_len+1) * (loop_params.gmres_restart_len+1)];
        double *R = new double[loop_params.gmres_restart_len * (loop_params.gmres_restart_len+1)];
        double *Q = new double[(loop_params.gmres_restart_len+1) * (loop_params.gmres_restart_len+1)]; // (m+1 x m+1)
        double *Q_copy = new double[(loop_params.gmres_restart_len+1) * (loop_params.gmres_restart_len+1)]; // (m+1 x m+1)
        double *g = new double[loop_params.gmres_restart_len+1];
        double *g_copy = new double[loop_params.gmres_restart_len+1];

        args->init_v = init_v;
        args->V = V;
        args->H = H;
        args->H_tmp = H_tmp;
        args->J = J;
        args->R = R;
        args->Q = Q;
        args->Q_copy = Q_copy;
        args->g = g;
        args->g_copy = g_copy;
    }
    
    args->coo_mat = coo_mat;
    args->x_star = x_star;
    args->x_old = x_old;
    args->x_new = x_new;
    args->tmp = tmp;
    args->D = D;
    args->r = r;
    args->b = b;
    args->normed_residuals = normed_residuals;

    args->loop_params = &loop_params;
    args->flags = &flags;
    args->matrix_file_name = &matrix_file_name;
    args->sparse_mat = sparse_mat;
    args->gmres_restart_len = loop_params.gmres_restart_len;

#ifdef __CUDACC__
    // Just give pointers to args struct now, allocate on device later
    double *d_x_star;
    double *d_x_new;
    double *d_x_old;
    int *d_row_ptr;
    int *d_col;
    double *d_val;
    double *d_tmp;
    double *d_r;
    double *d_D;
    double *d_b;
    double *d_normed_residuals;

    args->d_x_star = d_x_star;
    args->d_x_new = d_x_new;
    args->d_x_old = d_x_old;
    args->d_row_ptr = d_row_ptr;
    args->d_col = d_col;
    args->d_val = d_val;
    args->d_tmp = d_tmp;
    args->d_r = d_r;
    args->d_D = d_D;
    args->d_b = d_b;
    args->d_normed_residuals = d_normed_residuals;

    cudaMalloc(&(args->d_x_star), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_new), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_x_old), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_tmp), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_D), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_r), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_b), (args->vec_size)*sizeof(double));
    cudaMalloc(&(args->d_normed_residuals), (args->loop_params->max_iters / args->loop_params->residual_check_len + 1)*sizeof(double));
#endif

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

    solve(args, &loop_params);

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
    delete normed_residuals;
    delete x_star;
    delete x_old;
    delete x_new;
    delete tmp;
    delete D;
    delete r;
    delete b;

    if(args->solver_type == "gmres"){
        delete init_v;
        delete V;
        delete H;
        delete H_tmp;
        delete J;
        delete R;
        delete Q;
        delete Q_copy;
        delete g;
        delete g_copy;
    }

#ifdef __CUDACC__
    cudaFree(args->d_x_star);
    cudaFree(args->d_x_new);
    cudaFree(args->d_x_old);
    cudaFree(args->d_row_ptr);
    cudaFree(args->d_col);
    cudaFree(args->d_val);
    cudaFree(args->d_tmp);
    cudaFree(args->d_r);
    cudaFree(args->d_D);
    cudaFree(args->d_b);
    cudaFree(args->d_normed_residuals);
#endif

    return 0;
}
