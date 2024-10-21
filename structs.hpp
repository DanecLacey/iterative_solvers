#ifndef STRUCTS_H
#define STRUCTS_H

#ifdef USE_USPMV
#include "../Ultimate-SpMV/code/interface.hpp"
#endif

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

#include "stopwatch.hpp"
#include "kernels.hpp"


// NOTE: Is not having default bools really bad?
struct Flags
{
    bool print_iters;
    bool print_summary;
    bool print_residuals;
    bool convergence_flag;
    bool precondition;
    bool compare_direct;
    bool random_data;
};

// TODO: Make user-enetered option available
struct LoopParams
{
    int iter_count;
    int residual_count;
    int residual_check_len;
    int max_iters;
    double stopping_criteria;
    double tol;
    double init_b;
    double init_x;
    int gmres_restart_len;

    template <typename VT>
    double compute_stopping_criteria(
        std::string solver_type,
        VT *r,
        int n_cols
    ){
        // Precalculate stopping criteria
        // NOTE: Easier to always do on the host for now
        double norm_r0;
        double stopping_critera;

        if(solver_type == "gmres"){
            norm_r0 = euclidean_vec_norm_cpu(r, n_cols);
        }
        else{
            norm_r0 = infty_vec_norm_cpu(r, n_cols);
        }

        stopping_critera = this->tol * norm_r0;

#ifdef DEBUG_MODE
        printf("norm(initial residual) = %f\n", norm_r0);
        printf("stopping criteria = %f\n", stopping_criteria);
        std::cout << "stopping criteria = " << this->tol <<  " * " <<  norm_r0 << " = " << stopping_criteria << std::endl;
#endif

        return static_cast<double>(stopping_critera); 
    }
};

template <typename VT>
struct argType {
#ifdef USE_SCAMAC
    char *scamac_args;
#endif

#ifdef USE_AP
    double dp_percent;
    double sp_percent;
#ifdef HAVE_HALF_MATH
    double hp_percent;
#endif
#endif

    COOMtxData<double> *coo_mat;
    SparseMtxFormat<VT> *sparse_mat;
    Timers *timers;

    double *normed_residuals;
#ifdef __CUDACC__
    double *d_normed_residuals;
#endif
    LoopParams *loop_params;
    std::string solver_type;
    std::string preconditioner_type;
    std::string scale_type = "diag";
    Flags *flags;
    const std::string *matrix_file_name;
    double calc_time_elapsed;
    double total_time_elapsed;
    int vec_size;

    void allocate_cpu_general_structs()
    {
        std::cout << "Allocating space for general CPU structs" << std::endl;
        double *normed_residuals = new double[this->loop_params->max_iters / this->loop_params->residual_check_len + 1];
        this->normed_residuals = normed_residuals;
    }

#ifdef __CUDACC__
    template <typename VT>
    void allocate_gpu_general_structs()
    {
        std::cout << "Allocating space for general GPU structs" << std::endl;
        cudaMalloc(&(args->d_normed_residuals), (args->loop_params->max_iters / args->loop_params->residual_check_len + 1)*sizeof(double));     
    }
#endif

};

#endif /*STRUCTS_H*/