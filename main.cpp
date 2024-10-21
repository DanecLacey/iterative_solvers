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
#include "solver.hpp"
#include "solver_harness.hpp"
#include "structs.hpp"
#include "mmio.h"

#define xstr(s) str(s)
#define str(s) #s

template <size_t Length>
struct fixed_string {
    char _chars[Length+1] = {}; // +1 for null terminator
};

#ifdef USE_USPMV
template <typename VT>
void convert_ap_scs_structs(
    MtxData<double, int> *mtx_mat_dp,
    MtxData<float, int> *mtx_mat_sp,
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *mtx_mat_hp,
#endif
    argType<VT> *args,
    char *ap_value_type
){
    if(ap_value_type == "ap[dp_sp]"){
        std::cout << "Converting DP struct" << std::endl;
        convert_to_scs<double, double, int>(mtx_mat_dp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_dp); 
        std::cout << "Converting SP struct" << std::endl;
        convert_to_scs<float, float, int>(mtx_mat_sp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_sp, &(args->sparse_mat->scs_mat_dp->old_to_new_idx)[0]); 
        // Empty struct
#ifdef HAVE_HALF_MATH
        std::cout << "Converting HP struct" << std::endl;
        convert_to_scs<_Float16, _Float16, int>(mtx_mat_hp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_hp); 
#endif
    }
    else if(ap_value_type == "ap[dp_hp]"){
        std::cout << "Converting DP struct" << std::endl;
        convert_to_scs<double, double, int>(mtx_mat_dp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_dp); 
#ifdef HAVE_HALF_MATH
        std::cout << "Converting HP struct" << std::endl;
        convert_to_scs<_Float16, _Float16, int>(mtx_mat_hp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_hp, &(args->sparse_mat->scs_mat_dp->old_to_new_idx)[0]); 
#endif
        // Empty struct
        std::cout << "Converting SP struct" << std::endl;
        convert_to_scs<float, float, int>(mtx_mat_sp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_sp); 
    }
    else if(ap_value_type == "ap[sp_hp]"){
        std::cout << "Converting SP struct" << std::endl;
        convert_to_scs<float, float, int>(mtx_mat_sp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_sp); 

#ifdef HAVE_HALF_MATH
        std::cout << "Converting HP struct" << std::endl;
        convert_to_scs<_Float16, _Float16, int>(mtx_mat_hp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_hp, &(args->sparse_mat->scs_mat_sp->old_to_new_idx)[0]); 
#endif
        // Empty struct
        std::cout << "Converting DP struct" << std::endl;
        convert_to_scs<double, double, int>(mtx_mat_dp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_dp); 
    }
    else if(ap_value_type == "ap[dp_sp_hp]"){
        std::cout << "Converting DP struct" << std::endl;
        convert_to_scs<double, double, int>(mtx_mat_dp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_dp); 
        std::cout << "Converting SP struct" << std::endl;
        convert_to_scs<float, float, int>(mtx_mat_sp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_sp, &(args->sparse_mat->scs_mat_dp->old_to_new_idx)[0]); 

#ifdef HAVE_HALF_MATH
        std::cout << "Converting HP struct" << std::endl;
        convert_to_scs<_Float16, _Float16, int>(mtx_mat_hp, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat_hp, &(args->sparse_mat->scs_mat_dp->old_to_new_idx)[0]); 
#endif
    }
}
#endif

// TODO: MPI preprocessing will go here
template <typename VT>
void preprocessing(
    argType<VT> *args,
    Solver<VT> *solver
){

#ifdef OUTPUT_SPARSITY
    // Just to visualize sparsity
    std::string file_out_name = "output_matrix";
    std::cout << "Writing sparsity to mtx file..." << std::endl;
    args->coo_mat->write_to_mtx_file(0, file_out_name);
    std::cout << "Finished" << std::endl;
    exit(0);
#endif

    std::cout << "Preprocessing Matrix Data" << std::endl;
    timeval *preprocessing_start = new timeval;
    timeval *preprocessing_end = new timeval;
    Stopwatch *preprocessing_wtime = new Stopwatch(preprocessing_start, preprocessing_end);
    args->timers->preprocessing_wtime = preprocessing_wtime;
    args->timers->preprocessing_wtime->start_stopwatch();

    // Equilibration. An optional, but usually necessary preprocessing step for convergence
    std::vector<double> scaling_row_elems(args->coo_mat->n_cols, 1.0);
    std::vector<double> scaling_col_elems(args->coo_mat->n_cols, 1.0);

    if(args->scale_type == "max"){
        extract_largest_row_elems(args->coo_mat, &scaling_row_elems);
        extract_largest_col_elems(args->coo_mat, &scaling_row_elems);
        
    }
    else if(args->scale_type == "diag"){
        extract_diag<double>(args->coo_mat, scaling_row_elems.data(), true);
        extract_diag<double>(args->coo_mat, scaling_col_elems.data(), true);
    }

    scale_matrix_rows(args->coo_mat, &scaling_row_elems);
    scale_matrix_cols(args->coo_mat, &scaling_col_elems);


#ifdef USE_USPMV
    // Convert COO mat to Sell-C-Simga
    MtxData<double, int> *mtx_mat = new MtxData<double, int>;
    mtx_mat->n_rows = args->coo_mat->n_rows;
    mtx_mat->n_cols = args->coo_mat->n_cols;
    mtx_mat->nnz = args->coo_mat->nnz;
    mtx_mat->is_sorted = true; //TODO
    mtx_mat->is_symmetric = false; //TODO
    mtx_mat->I = args->coo_mat->I;
    mtx_mat->J = args->coo_mat->J;
    mtx_mat->values = args->coo_mat->values;

    convert_to_scs<double, VT, int>(mtx_mat, CHUNK_SIZE, SIGMA, args->sparse_mat->scs_mat);

    // NOTE: We change vec_size here, so all structs from here on will be different size!
    args->vec_size = args->sparse_mat->scs_mat->n_rows_padded;
#else
    args->vec_size = args->coo_mat->n_cols;
#endif

    // Only now that we know args->vec_size, we can allocate structs 
    args->allocate_cpu_general_structs();

#ifdef __CUDACC__
    args->allocate_gpu_general_structs();
#endif

#ifdef USE_USPMV
#ifdef USE_AP
    // Convert MTX mat to Sell-C-Simga in the case of AP
    MtxData<double, int> *mtx_mat_dp = new MtxData<double, int>;
    MtxData<float, int> *mtx_mat_sp = new MtxData<float, int>;
#ifdef HAVE_HALF_MATH
    MtxData<_Float16, int> *mtx_mat_hp = new MtxData<_Float16, int>;
#endif

    partition_precisions<double, int>(
        mtx_mat, 
        mtx_mat_dp, 
        mtx_mat_sp, 
#ifdef HAVE_HALF_MATH
        mtx_mat_hp, 
#endif
        &scaling_row_elems, 
        &scaling_col_elems, 
        AP_THRESHOLD_1,
        AP_THRESHOLD_2,
        AP_VALUE_TYPE,
        true
    );

    args->dp_percent = mtx_mat_dp->nnz / (double)mtx_mat->nnz;
    args->sp_percent = mtx_mat_sp->nnz / (double)mtx_mat->nnz;
#ifdef HAVE_HALF_MATH
    args->hp_percent = mtx_mat_hp->nnz / (double)mtx_mat->nnz;
#endif

    convert_ap_scs_structs<VT>(
        mtx_mat_dp,
        mtx_mat_sp,
#ifdef HAVE_HALF_MATH
        mtx_mat_hp,
#endif
        args,
        AP_VALUE_TYPE
    );

#endif
#endif

    // Allocate structs specific to each solver
    solver->allocate_cpu_solver_structs(
#ifdef USE_AP
        mtx_mat_dp,
        mtx_mat_sp,
#ifdef HAVE_HALF_MATH
        mtx_mat_hp,
#endif
        AP_VALUE_TYPE,
#endif
        args->sparse_mat, 
        args->coo_mat, 
        args->vec_size
    );

//     // Just convenient to have a CRS copy too
    convert_to_crs<double, VT>(
        &args->coo_mat->n_rows,
        &args->coo_mat->n_cols,
        &args->coo_mat->nnz,
        &args->coo_mat->I,
        &args->coo_mat->J,
        &args->coo_mat->values, 
        args->sparse_mat->crs_mat
    );
#ifdef USE_USPMV
#ifdef USE_AP
    convert_to_crs<double, double>(
        &args->coo_mat->n_rows,
        &args->coo_mat->n_cols,
        &args->coo_mat->nnz,
        &args->coo_mat->I,
        &args->coo_mat->J,
        &args->coo_mat->values, 
        args->sparse_mat->crs_mat_dp
    );
    convert_to_crs<double, float>(
        &args->coo_mat->n_rows,
        &args->coo_mat->n_cols,
        &args->coo_mat->nnz,
        &args->coo_mat->I,
        &args->coo_mat->J,
        &args->coo_mat->values, 
        args->sparse_mat->crs_mat_sp
    );
#ifdef HAVE_HALF_MATH
    convert_to_crs<double, _Float16>(
        &args->coo_mat->n_rows,
        &args->coo_mat->n_cols,
        &args->coo_mat->nnz,
        &args->coo_mat->I,
        &args->coo_mat->J,
        &args->coo_mat->values, 
        args->sparse_mat->crs_mat_hp
    );
#endif
#endif
#endif

#ifdef __CUDACC__
    gpu_allocate_copy_sparse_mat(args);
#endif

    extract_diag<VT>(args->coo_mat, solver->D);

    // Make b vector
    generate_vector<VT>(solver->b, args->vec_size, args->flags->random_data, &(args->coo_mat->values)[0], args->loop_params->init_b);
    scale_vector<VT>(solver->b, &scaling_row_elems, args->vec_size);

    // Make initial x vector
    generate_vector<VT>(solver->x_old, args->vec_size, args->flags->random_data, &(args->coo_mat->values)[0], args->loop_params->init_x);
    scale_vector<VT>(solver->x_old, &scaling_row_elems, args->vec_size);

    

    // TODO: Ugly, wrap-up in a subroutine
#ifdef USE_AP
    // Copy data to AP structs
    std::string working_precision = xstr(WORKING_PRECISION);
    if(working_precision == "double"){
        for(int i = 0; i < args->vec_size; ++i){
            solver->x_old_dp[i] = solver->x_old[i];
        }
    }
    else if(working_precision == "float"){
        for(int i = 0; i < args->vec_size; ++i){
            solver->x_old_sp[i] = solver->x_old[i];
        }
    }
#ifdef HAVE_HALF_MATH
    else if(working_precision == "half"){
        for(int i = 0; i < args->vec_size; ++i){
            solver->x_old_hp[i] = solver->x_old[i];
        }
    }
#endif
#endif

#ifdef __CUDACC__
    gpu_copy_structs(args);
#endif

    solver->init_residual(args->sparse_mat, args->coo_mat->n_cols, args->vec_size);

#ifdef DEBUG_MODE
    printf("initial residual = [");
    for(int i = 0; i < args->coo_mat->n_cols; ++i){
        std::cout << static_cast<double>(solver->r[i]) << ",";
    }
    printf("]\n");
#endif

    solver->init_structs(args->sparse_mat, args->coo_mat, args->timers, args->vec_size);

    args->loop_params->stopping_criteria = args->loop_params->template compute_stopping_criteria<VT>(args->solver_type, solver->r, args->coo_mat->n_cols);

// #ifdef __CUDACC__
//     // The first residual is computed on the host, and given to the device
//     // Easier to just do on the host for now, and give stopping criteria to device
//     cudaMalloc(&(args->loop_params->d_stopping_criteria), sizeof(double));
//     cudaMemcpy(args->loop_params->d_stopping_criteria, &(args->loop_params->stopping_criteria), sizeof(double), cudaMemcpyHostToDevice);
// #endif

    args->timers->preprocessing_wtime->end_stopwatch();
}

template <typename VT>
void run_solver(
    int argc, 
    char *argv[],
    Flags *flags, 
    LoopParams *loop_params
){

    Timers *timers = new Timers;
    timeval *total_time_start = new timeval;
    timeval *total_time_end = new timeval;
    Stopwatch *total_wtime = new Stopwatch(total_time_start, total_time_end);
    std::string matrix_file_name;
    SparseMtxFormat<VT> *sparse_mat = new SparseMtxFormat<VT>;
    COOMtxData<double> *coo_mat = new COOMtxData<double>;
    argType<VT> *args = new argType<VT>;

    args->timers = timers;
    args->timers->total_wtime = total_wtime;
    args->timers->total_wtime->start_stopwatch();

    assign_cli_inputs<VT>(
        args, 
        argc, 
        argv,
#ifdef USE_AP
        AP_VALUE_TYPE,
#endif 
        &matrix_file_name
    );

    Solver<VT> *solver = new Solver<VT>(args->solver_type, args->preconditioner_type);
    gmresArgs<VT> *gmres_args = new gmresArgs<VT>;
    cgArgs<VT> *cg_args = new cgArgs<VT>;
    solver->gmres_args = gmres_args;
    solver->cg_args = cg_args;

#ifdef USE_SCAMAC
        std::cout << "Generating Matrix" << std::endl;
        matrix_file_name = args->scamac_args;
        scamac_make_mtx(args, coo_mat);
#else
        std::cout << "Reading Matrix" << std::endl;
        read_mtx(matrix_file_name, coo_mat);
#endif

        args->loop_params = loop_params;
        args->flags = flags;
        args->matrix_file_name = &matrix_file_name;
        args->sparse_mat = sparse_mat;
        solver->gmres_args->restart_length = loop_params->gmres_restart_len;
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

        solver->d_x_star = d_x_star;
        solver->d_x_new = d_x_new;
        solver->d_x_old = d_x_old;
        solver->d_row_ptr = d_row_ptr;
        solver->d_col = d_col;
        solver->d_val = d_val;
        solver->d_tmp = d_tmp;
        solver->d_r = d_r;
        solver->d_D = d_D;
        solver->d_b = d_b;
        args->d_normed_residuals = d_normed_residuals;
#endif

#ifdef USE_USPMV
        ScsData<VT, int> *scs_mat = new ScsData<VT, int>;
        args->sparse_mat->scs_mat = scs_mat;
#ifdef USE_AP
        ScsData<double, int> *scs_mat_dp = new ScsData<double, int>;
        ScsData<float, int> *scs_mat_sp = new ScsData<float, int>;
#ifdef HAVE_HALF_MATH
        ScsData<_Float16, int> *scs_mat_hp = new ScsData<_Float16, int>;
#endif

        args->sparse_mat->scs_mat_dp = scs_mat_dp;
        args->sparse_mat->scs_mat_sp = scs_mat_sp;
#ifdef HAVE_HALF_MATH
        args->sparse_mat->scs_mat_hp = scs_mat_hp;
#endif

#endif
        ScsData<VT, int> *scs_L = new ScsData<VT, int>;
        ScsData<VT, int> *scs_U = new ScsData<VT, int>;
        
        args->sparse_mat->scs_L = scs_L;
        args->sparse_mat->scs_U = scs_U;

#ifdef USE_AP
        ScsData<double, int> *scs_L_dp = new ScsData<double, int>;
        ScsData<double, int> *scs_U_dp = new ScsData<double, int>;
        ScsData<float, int> *scs_L_sp = new ScsData<float, int>;
        ScsData<float, int> *scs_U_sp = new ScsData<float, int>;
        
        args->sparse_mat->scs_L_dp = scs_L_dp;
        args->sparse_mat->scs_U_dp = scs_U_dp;
        args->sparse_mat->scs_L_sp = scs_L_sp;
        args->sparse_mat->scs_U_sp = scs_U_sp;
#ifdef HAVE_HALF_MATH
        ScsData<_Float16, int> *scs_L_hp = new ScsData<_Float16, int>;
        ScsData<_Float16, int> *scs_U_hp = new ScsData<_Float16, int>;
        
        args->sparse_mat->scs_L_hp = scs_L_hp;
        args->sparse_mat->scs_U_hp = scs_U_hp;
#endif

#endif

#endif

        // TODO: Put this somewhere other than main routine
        CRSMtxData<VT> *crs_mat = new CRSMtxData<VT>;
        CRSMtxData<VT> *crs_L = new CRSMtxData<VT>;
        CRSMtxData<VT> *crs_U = new CRSMtxData<VT>;
        args->sparse_mat->crs_mat = crs_mat;
        args->sparse_mat->crs_L = crs_L;
        args->sparse_mat->crs_U = crs_U;
#ifdef USE_USPMV
#ifdef USE_AP
        CRSMtxData<double> *crs_mat_dp = new CRSMtxData<double>;
        CRSMtxData<double> *crs_L_dp =   new CRSMtxData<double>;
        CRSMtxData<double> *crs_U_dp =   new CRSMtxData<double>;
        args->sparse_mat->crs_mat_dp = crs_mat_dp;
        args->sparse_mat->crs_L_dp =   crs_L_dp;
        args->sparse_mat->crs_U_dp =   crs_U_dp;
        CRSMtxData<float> *crs_mat_sp = new CRSMtxData<float>;
        CRSMtxData<float> *crs_L_sp =   new CRSMtxData<float>;
        CRSMtxData<float> *crs_U_sp =   new CRSMtxData<float>;
        args->sparse_mat->crs_mat_sp = crs_mat_sp;
        args->sparse_mat->crs_L_sp =   crs_L_sp;
        args->sparse_mat->crs_U_sp =   crs_U_sp;
#ifdef HAVE_HALF_MATH
        CRSMtxData<_Float16> *crs_mat_hp = new CRSMtxData<_Float16>;
        CRSMtxData<_Float16> *crs_L_hp =   new CRSMtxData<_Float16>;
        CRSMtxData<_Float16> *crs_U_hp =   new CRSMtxData<_Float16>;
        args->sparse_mat->crs_mat_hp = crs_mat_hp;
        args->sparse_mat->crs_L_hp =   crs_L_hp;
        args->sparse_mat->crs_U_hp =   crs_U_hp;
#endif
#endif
#endif
        preprocessing<VT>(args, solver);

        solve<VT>(args, solver);

        postprocessing<VT>(args, solver);

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
        1.0, // init value for x
        GMRES_RESTART_LEN // GMRES restart length
    };
////////////////////////////////////////////////

    std::string working_precision = xstr(WORKING_PRECISION);
    if(working_precision == "double")
        run_solver<double>(argc, argv, &flags, &loop_params);
    else if(working_precision == "float")
        run_solver<float>(argc, argv, &flags, &loop_params);
    else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
        run_solver<_Float16>(argc, argv, &flags, &loop_params);
#endif
    }
    else{
        std::cout << "PRECISION \"" << working_precision << "\" in config.mk not recognized" << std::endl;
    }

#ifdef USE_LIKWID
    LIKWID_MARKER_CLOSE;
#endif

    return 0;
}
