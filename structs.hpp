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
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"


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
};

class Solver
{
    // Method-specific functions come from methods/ dir
    /*
    Data: 
        x
        x_old
        x_new
        all gmres args
    Methods:
        get x
        print x
        calc residual
        init structs
        init timers
        register likwid markers
        iterate
        record residual norm
    */ 
   std::string solver_type;

public:
    double *x_star;
    double *x_new;
    double *x_old;
    double *x;
    double *tmp;
    double *D;
    double *r;
    double *b;

    gmresArgs *gmres_args;

#ifdef __CUDACC__
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
#endif

    Solver(std::string _solver_type) : solver_type(_solver_type) {}

    // Put common methods out here
    void copy_fresh_x(
        double *fresh_x,
        double *fresh_x_new,
        double *fresh_x_old,
        int N
    ){
        // Need deep copy TODO: rethink and do better
        #pragma omp parallel for
        for(int i = 0; i < N; ++i){
            fresh_x[i] = this->x_old[i];
            fresh_x_new[i] = this->x_new[i];
            fresh_x_old[i] = this->x_old[i];
        }

        this->x = fresh_x;
        this->x_new = fresh_x_new;
        this->x_old = fresh_x_old;       
    }

    void allocate_structs(
        SparseMtxFormat *sparse_mat,
        COOMtxData *coo_mat,
        Timers *timers,
        int vec_size
    ){
        if(solver_type == "jacobi"){
            
        }
        else if(solver_type == "gauss-seidel"){
            
        }
        else if(solver_type == "gmres"){
            allocate_gmres_structs(this->gmres_args, vec_size);
        }
    }

    void init_structs(
        SparseMtxFormat *sparse_mat,
        COOMtxData *coo_mat,
        Timers *timers,
        int vec_size
    ){
        if(solver_type == "jacobi"){

        }
        else if(solver_type == "gauss-seidel"){
            init_gs_structs(coo_mat, sparse_mat);
        }
        else if(solver_type == "gmres"){
            init_gmres_structs(this->gmres_args, this->r, coo_mat->n_rows);
            init_gmres_timers(timers);
        }
    }

#ifdef USE_LIKWID
    void register_likwid_markers(){
        #pragma omp parallel
        {
#ifdef USE_USPMV
#ifdef USE_AP
            LIKWID_MARKER_REGISTER("uspmv_ap_crs_benchmark");
#else
            LIKWID_MARKER_REGISTER("uspmv_crs_benchmark");
#endif

#else 
            LIKWID_MARKER_REGISTER("native_spmv_benchmark");
#endif
        }
    }
#endif

    void iterate(
        SparseMtxFormat *sparse_mat,
        Timers *timers,
        int vec_size,
        int n_rows,
        int iter_count,
        double *residual_norm
    ){
        if(solver_type == "jacobi"){
            jacobi_iteration_sep_cpu(
                sparse_mat,
                this->D,
                this->b,
                this->x_old,
                this->x_new,
                n_rows
            );
        }
        else if(solver_type == "gauss-seidel"){
            gs_iteration_sep_cpu(
                sparse_mat, 
                this->tmp, 
                this->D, 
                this->b, 
                this->x,
                n_rows
            );
        }
        else if(solver_type == "gmres"){
            gmres_iteration_ref_cpu(
                sparse_mat, 
                timers,
                this->gmres_args->V,
                this->gmres_args->H,
                this->gmres_args->H_tmp,
                this->gmres_args->J,
                this->gmres_args->Q,
                this->gmres_args->Q_copy,
                this->tmp, 
                this->gmres_args->R,
                this->gmres_args->g,
                this->gmres_args->g_copy,
                this->b, 
                this->x,
                this->gmres_args->beta,
                vec_size,
                this->gmres_args->restart_count,
                iter_count,
                residual_norm,
                this->gmres_args->restart_length
            );
        }
    }

    void restart_gmres(
        Timers *timers,
        SparseMtxFormat *sparse_mat,
        int vec_size,
        int n_cols,
        int iter_count
    ){
#ifdef DEBUG_MODE
        std::cout << "RESTART GMRES" << std::endl;
#endif
        timers->gmres_get_x_wtime->start_stopwatch();
        gmres_get_x(
            this->gmres_args->R, 
            this->gmres_args->g, 
            this->x, 
            this->x_old, 
            this->gmres_args->V, 
            this->gmres_args->Vy, 
            vec_size, 
            this->gmres_args->restart_count, 
            iter_count, 
            this->gmres_args->restart_length
        );
        timers->gmres_get_x_wtime->end_stopwatch();

        calc_residual_cpu(sparse_mat, this->x, this->b, this->r, this->tmp, vec_size);

    #ifdef DEBUG_MODE
        printf("restart residual = [");
        for(int i = 0; i < n_cols; ++i){
            std::cout << this->r[i] << ",";
        }
        printf("]\n");
    #endif

        this->gmres_args->beta = euclidean_vec_norm_cpu(this->r, vec_size); 
        scale(this->gmres_args->init_v, this->r, 1 / this->gmres_args->beta, vec_size);

    #ifdef DEBUG_MODE
        std::cout << "Restarted Beta = " << this->gmres_args->beta << std::endl;          

        // TODO: Is this vec_size, or n_cols?
        std::cout << "init_v = [";
            for(int i = 0; i < vec_size; ++i){
                std::cout << this->gmres_args->init_v[i] << ", ";
            }
        std::cout << "]" << std::endl;
    #endif
        double norm_r0 = euclidean_vec_norm_cpu(this->r, vec_size);

    #ifdef DEBUG_MODE
        printf("restarted norm(initial residual) = %f\n", norm_r0);
    #endif

        init_gmres_structs(this->gmres_args, this->r, vec_size);
        ++this->gmres_args->restart_count;

    #ifdef DEBUG_MODE
        std::cout << "Restarted GMRES outputting the x vector [" << std::endl;
        for(int i = 0; i < vec_size; ++i){
            std::cout << this->x[i] << std::endl;
        }
        std::cout << "]" << std::endl;
    #endif

        // Need deep copy TODO: rethink and do better
        #pragma omp parallel for
        for(int i = 0; i < vec_size; ++i){
            this->x_old[i] = this->x[i];
        }
    }

    void save_x_star(
        Timers *timers,
        int vec_size,
        int iter_count
    ){
        if(solver_type == "jacobi"){
            // TODO: Bandaid, since swapping seems to not work :(
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star[i] = this->x_old[i];
#ifdef DEBUG_MODE
                std::cout << "args->x_star[" << i << "] = " << this->x_old[i] << "==" << this->x_star[i] << std::endl;
#endif
            }
            // std::swap(args->x_star, x_old);

        }
        else if (solver_type == "gauss-seidel"){
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star[i] = this->x[i];
#ifdef DEBUG_MODE
                std::cout << "args->x_star[" << i << "] = " << this->x[i] << "==" << this->x_star[i] << std::endl;
#endif
            }
            // std::swap(x, args->x_star);
        }    
        else if(solver_type == "gmres"){
            timers->gmres_get_x_wtime->start_stopwatch();
            gmres_get_x(
                this->gmres_args->R, 
                this->gmres_args->g, 
                this->x_star, 
                this->x_old, 
                this->gmres_args->V,
                this->gmres_args->Vy, 
                vec_size, 
                this->gmres_args->restart_count, 
                iter_count - 1, 
                this->gmres_args->restart_length
            );

            timers->gmres_get_x_wtime->end_stopwatch();
        }
    }

    void exchange_arrays(int vec_size){
        if(solver_type == "jacobi"){
            for(int i = 0; i < vec_size; ++i){
                this->x_old[i] = this->x_new[i];
            }
        }
    }

#ifdef USE_USPMV
    void unpermute_x_star(
        int vec_size,
        int n_cols,
        int *old_to_new_idx
    ){
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[vec_size];
    apply_permutation(x_star_perm, this->x_star, old_to_new_idx, vec_size);

    // Deep copy, so you can free memory
    // NOTE: You do not take SCS padding with you! 
    #pragma omp parallel for
    for(int i = 0; i < n_cols; ++i){
        this->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
    }
#endif

};

class Preconditioner
{
    /*
        TODO
    */

};

struct argType {
#ifdef USE_SCAMAC
    char *scamac_args;
#endif

#ifdef USE_AP
    double lp_percent;
    double hp_percent;
#endif

    COOMtxData *coo_mat;
    SparseMtxFormat *sparse_mat;
    Timers *timers;
    Solver *solver;
    Preconditioner *preconditioner;

    double *normed_residuals;
#ifdef __CUDACC__
    double *d_normed_residuals;
#endif
    LoopParams *loop_params;
    std::string solver_type;
    Flags *flags;
    const std::string *matrix_file_name;
    double calc_time_elapsed;
    double total_time_elapsed;
    int vec_size;
};

#endif /*STRUCTS_H*/