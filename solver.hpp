#ifndef SOLVER_HPP
#define SOLVER_HPP

#include "stopwatch.hpp"
#include "sparse_matrix.hpp"
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"
#include "methods/cg.hpp"

#define xstr(s) str(s)
#define str(s) #s

// Nothing here can reply on utility funcs
template <typename VT>
class Solver
{
   std::string solver_type;
   std::string preconditioner_type;

public:
    // Used for pure precision kernels 
    VT *x_star;
    VT *x_new;
    VT *x_old;
    VT *x;
    VT *x_perm;
    VT *x_old_perm;
    VT *x_new_perm;
    VT *tmp;
    VT *tmp_perm;
    VT *D;
    VT *r;
    VT *b;

    // Used for multi precision kernels
#ifdef USE_AP
    double *x_star_dp;
    double *x_new_dp;
    double *x_old_dp;
    double *x_dp;
    double *x_perm_dp;
    double *x_old_perm_dp;
    double *x_new_perm_dp;
    double *tmp_dp;
    double *tmp_perm_dp;

    float *x_star_sp;
    float *x_new_sp;
    float *x_old_sp;
    float *x_sp;
    float *x_perm_sp;
    float *x_old_perm_sp;
    float *x_new_perm_sp;
    float *tmp_sp;
    float *tmp_perm_sp;
#ifdef HAVE_HALF_MATH
    _Float16 *x_star_hp;
    _Float16 *x_new_hp;
    _Float16 *x_old_hp;
    _Float16 *x_hp;
    _Float16 *x_perm_hp;
    _Float16 *x_old_perm_hp;
    _Float16 *x_new_perm_hp;
    _Float16 *tmp_hp;
    _Float16 *tmp_perm_hp;
#endif
#endif

    gmresArgs<VT> *gmres_args;
    cgArgs<VT> *cg_args;

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

    Solver(std::string _solver_type, std::string _preconditioner_type) : solver_type(_solver_type), preconditioner_type(_preconditioner_type) {}

    template <typename T>
    void apply_residual_preconditioner(
        std::string preconditioner_type,
        SparseMtxFormat<VT> *sparse_mat,
        T *vec,
        T *rhs,
        VT *D,
        int n_cols){
        if(!preconditioner_type.empty()){
            if(preconditioner_type == "jacobi"){
                // TODO
            }
            else if(preconditioner_type == "gauss-seidel"){
                spltsv_crs<T, VT>(sparse_mat->crs_L, vec, D, rhs);
            }
            else{
                printf("ERROR: apply_residual_preconditioner: preconditioner_type not recognized.\n");
                exit(1);
            }
        }
    }

#ifdef __CUDACC__
    void allocate_gpu_solver_structs(
        int vec_size,
        int n_cols
    ){
        cudaMalloc(&(this->d_x_star), (n_cols)*sizeof(double));
        cudaMalloc(&(this->d_x_new), (vec_size)*sizeof(double));
        cudaMalloc(&(this->d_x_old), (vec_size)*sizeof(double));
        cudaMalloc(&(this->d_tmp), (n_cols)*sizeof(double));
        cudaMalloc(&(this->d_D), (n_cols)*sizeof(double));
        cudaMalloc(&(this->d_r), (n_cols)*sizeof(double));
        cudaMalloc(&(this->d_b), (n_cols)*sizeof(double));
    }

    void copy_gpu_structs(
        int vec_size,
        int n_cols
    ){
        cudaMemcpy(this->d_x_star, this->x_star, n_cols*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_x_new, this->x_new, vec_size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_x_old, this->x_old, vec_size*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_tmp, this->tmp, n_cols*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_D, this->D, n_cols*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_r, this->r, n_cols*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_b, this->b, n_cols*sizeof(double), cudaMemcpyHostToDevice);
    }

    // TODO: IDK if this should be in "solver"
    template <typename VT>
    void allocate_copy_gpu_sparse_mat(
        argType<VT> *args
    ){
        cudaMalloc(&(this->d_row_ptr), (args->sparse_mat->crs_mat->n_rows+1)*sizeof(int));
        cudaMalloc(&(this->d_col), (args->sparse_mat->crs_mat->nnz)*sizeof(int));
        cudaMalloc(&(this->d_val), (args->sparse_mat->crs_mat->nnz)*sizeof(double));
        cudaMemcpy(this->d_row_ptr, &(args->sparse_mat->crs_mat->row_ptr)[0], (args->sparse_mat->crs_mat->n_rows+1)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_col, &(args->sparse_mat->crs_mat->col)[0], (args->sparse_mat->crs_mat->nnz)*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_val, &(args->sparse_mat->crs_mat->val)[0], (args->sparse_mat->crs_mat->nnz)*sizeof(double), cudaMemcpyHostToDevice);
    }
#endif

    void allocate_cpu_solver_structs(
#ifdef USE_AP
        MtxData<double, int> *mtx_mat_dp,
        MtxData<float, int> *mtx_mat_sp,
#ifdef HAVE_HALF_MATH
        MtxData<_Float16, int> *mtx_mat_hp,
#endif
        char* ap_value_type,
#endif
        SparseMtxFormat<VT> *sparse_mat,
        COOMtxData<double> *coo_mat,
        int vec_size
    ){
        int n_cols = coo_mat->n_cols;
        std::cout << "Allocating space for Solver structs" << std::endl;

        // Pure precision
        VT *x_star = new VT[n_cols];
        VT *x = new VT[vec_size];
        VT *x_old = new VT[vec_size];
        VT *x_new = new VT[vec_size];
    #ifdef USE_USPMV
        VT *x_perm = new VT[vec_size];
        VT *x_old_perm = new VT[vec_size];
        VT *x_new_perm = new VT[vec_size];
    #endif
        VT *tmp = new VT[n_cols];
        VT *tmp_perm = new VT[n_cols];
        VT *D = new VT[n_cols];
        VT *r = new VT[n_cols];
        VT *b = new VT[n_cols];

        // Adaptive precision
#ifdef USE_USPMV
#ifdef USE_AP
        double *x_star_dp = new double[n_cols];
        double *x_dp = new double[vec_size];
        double *x_old_dp = new double[vec_size];
        double *x_new_dp = new double[vec_size];
        double *x_perm_dp = new double[vec_size];
        double *x_old_perm_dp = new double[vec_size];
        double *x_new_perm_dp = new double[vec_size];
        double *tmp_dp = new double[n_cols];
        double *tmp_perm_dp = new double[n_cols];

        float *x_star_sp = new float[n_cols];
        float *x_sp = new float[vec_size];
        float *x_old_sp = new float[vec_size];
        float *x_new_sp = new float[vec_size];
        float *x_perm_sp = new float[vec_size];
        float *x_old_perm_sp = new float[vec_size];
        float *x_new_perm_sp = new float[vec_size];
        float *tmp_sp = new float[n_cols];
        float *tmp_perm_sp = new float[n_cols];

#ifdef HAVE_HALF_MATH
        _Float16 *x_star_hp = new _Float16[n_cols];
        _Float16 *x_hp = new _Float16[vec_size];
        _Float16 *x_old_hp = new _Float16[vec_size];
        _Float16 *x_new_hp = new _Float16[vec_size];
        _Float16 *x_perm_hp = new _Float16[vec_size];
        _Float16 *x_old_perm_hp = new _Float16[vec_size];
        _Float16 *x_new_perm_hp = new _Float16[vec_size];
        _Float16 *tmp_hp = new _Float16[n_cols];
        _Float16 *tmp_perm_hp = new _Float16[n_cols];
#endif
#endif
#endif

        // Just set everything to zero here to avoid numerical/memory problems later
        #pragma omp parallel for
        for(int i = 0; i < vec_size; ++i){
            x[i] =             (VT)(0.0);
            x_old[i] =         (VT)(0.0);
            x_new[i] =         (VT)(0.0);
#ifdef USE_USPMV
            x_perm[i] =        (VT)(0.0);
            x_old_perm[i] =    (VT)(0.0);
            x_new_perm[i] =    (VT)(0.0);
#ifdef USE_AP
            x_dp[i] =          0.0;
            x_old_dp[i] =      0.0;
            x_new_dp[i] =      0.0;
            x_perm_dp[i] =     0.0;
            x_old_perm_dp[i] = 0.0;
            x_new_perm_dp[i] = 0.0;

            x_sp[i] =          0.0f;
            x_old_sp[i] =      0.0f;
            x_new_sp[i] =      0.0f;
            x_perm_sp[i] =     0.0f;
            x_old_perm_sp[i] = 0.0f;
            x_new_perm_sp[i] = 0.0f;
#ifdef HAVE_HALF_MATH
            x_hp[i] =          0.0f16;
            x_old_hp[i] =      0.0f16;
            x_new_hp[i] =      0.0f16;
            x_perm_hp[i] =     0.0f16;
            x_old_perm_hp[i] = 0.0f16;
            x_new_perm_hp[i] = 0.0f16;
#endif
#endif
#endif
        }

        #pragma omp parallel for
        for(int i = 0; i < n_cols; ++i){
            x_star[i] =    (VT)(0.0);
            tmp[i] =       (VT)(0.0);
            tmp_perm[i] =  (VT)(0.0);
            D[i] =         (VT)(0.0);
            r[i] =         (VT)(0.0);
            b[i] =         (VT)(0.0);
#ifdef USE_USPMV
#ifdef USE_AP
            x_star_dp[i] =     0.0;
            tmp_dp[i] =        0.0;
            tmp_perm_dp[i] =   0.0;

            x_star_sp[i] =     0.0f;
            tmp_sp[i] =        0.0f;
            tmp_perm_sp[i] =   0.0f;
#ifdef HAVE_HALF_MATH
            x_star_hp[i] =     0.0f16;
            tmp_hp[i] =        0.0f16;
            tmp_perm_hp[i] =   0.0f16;
#endif
#endif
#endif
        }

        this->x = x;
        this->x_star = x_star;
        this->x_old = x_old;
        this->x_new = x_new;
#ifdef USE_USPMV
        this->x_perm = x_perm;
        this->x_old_perm = x_old_perm;
        this->x_new_perm = x_new_perm;
#endif
        this->tmp = tmp;
        this->tmp_perm = tmp_perm;
        this->D = D;
        this->r = r;
        this->b = b;

#ifdef USE_USPMV
#ifdef USE_AP
        this->x_dp = x_dp;
        this->x_star_dp = x_star_dp;
        this->x_old_dp = x_old_dp;
        this->x_new_dp = x_new_dp;
        this->x_perm_dp = x_perm_dp;
        this->x_old_perm_dp = x_old_perm_dp;
        this->x_new_perm_dp = x_new_perm_dp;
        this->tmp_dp = tmp_dp;
        this->tmp_perm_dp = tmp_perm_dp;
        
        this->x_sp = x_sp;
        this->x_star_sp = x_star_sp;
        this->x_old_sp= x_old_sp;
        this->x_new_sp= x_new_sp;
        this->x_perm_sp = x_perm_sp;
        this->x_old_perm_sp = x_old_perm_sp;
        this->x_new_perm_sp = x_new_perm_sp;
        this->tmp_sp = tmp_sp;
        this->tmp_perm_sp = tmp_perm_sp;

#ifdef HAVE_HALF_MATH
        this->x_hp = x_hp;
        this->x_star_hp = x_star_hp;
        this->x_old_hp = x_old_hp;
        this->x_new_hp = x_new_hp;
        this->x_perm_hp = x_perm_hp;
        this->x_old_perm_hp = x_old_perm_hp;
        this->x_new_perm_hp = x_new_perm_hp;
        this->tmp_hp = tmp_hp;
        this->tmp_perm_hp = tmp_perm_hp;
#endif
#endif
#endif

        // Allocate Solver structs

        if(solver_type == "jacobi"){
            // TODO ?
        }
        else if(solver_type == "gauss-seidel"){
            allocate_gs_structs<VT>(
#ifdef USE_AP
                mtx_mat_dp,
                mtx_mat_sp,
#ifdef HAVE_HALF_MATH
                mtx_mat_hp,
#endif
                ap_value_type,
#endif
                coo_mat, 
                sparse_mat
            );
        }
        else if(solver_type == "gmres"){
            allocate_gmres_structs(this->gmres_args, vec_size);
        }
        else if(solver_type == "conjugate-gradient"){
            allocate_cg_structs(this->cg_args, n_cols); // <- or vec_size?
        }

        // Allocate Preconditioner structs
        if(!this->preconditioner_type.empty()){
            if(preconditioner_type == "jacobi"){
                // TODO
            }
            else if(preconditioner_type == "gauss-seidel"){
                allocate_gs_structs<VT>(
#ifdef USE_AP
                    mtx_mat_dp,
                    mtx_mat_sp,
#ifdef HAVE_HALF_MATH
                    mtx_mat_hp,
#endif
                    ap_value_type,
#endif
                    coo_mat, 
                    sparse_mat
                );
            }
        }
    }

    void init_structs(
        SparseMtxFormat<VT> *sparse_mat,
        COOMtxData<double> *coo_mat,
        Timers *timers,
        int vec_size
    ){
        if(solver_type == "jacobi"){

        }
        else if(solver_type == "gauss-seidel"){
            // init_gs_structs<VT>(coo_mat, sparse_mat);
        }
        else if(solver_type == "gmres"){
            init_gmres_structs(this->gmres_args, this->r, coo_mat->n_rows);
            init_gmres_timers(timers);
        }
        else if(solver_type == "conjugate-gradient"){
            init_cg_structs(this->cg_args, this->r, coo_mat->n_rows);
            init_cg_timers(timers);
        }
    }

    void iterate(
        SparseMtxFormat<VT> *sparse_mat,
        Timers *timers,
        int vec_size,
        int n_rows,
        int iter_count,
        VT *residual_norm
    ){
        if(solver_type == "jacobi"){
            jacobi_iteration_sep_cpu<VT>(
                sparse_mat,
                this->D,
                this->b,
                this->x_old,
                this->x_old_perm,
                this->x_new,
                this->x_new_perm,
#ifdef USE_USPMV
#ifdef USE_AP
                this->x_old_dp,
                this->x_old_perm_dp,
                this->x_new_dp,
                this->x_new_perm_dp,
                this->x_old_sp,
                this->x_old_perm_sp,
                this->x_new_sp,
                this->x_new_perm_sp,
#ifdef HAVE_HALF_MATH
                this->x_old_hp,
                this->x_old_perm_hp,
                this->x_new_hp,
                this->x_new_perm_hp,
#endif
#endif
#endif
                n_rows
            );
        }
        else if(solver_type == "gauss-seidel"){
            gs_iteration_sep_cpu<VT>(
                sparse_mat, 
                this->tmp, 
                this->tmp_perm, 
                this->D, 
                this->b, 
                this->x,
#ifdef USE_USPMV
#ifdef USE_AP
                this->tmp_dp,
                this->tmp_perm_dp,
                this->x_dp,
                this->tmp_sp,
                this->tmp_perm_sp,
                this->x_sp,
#ifdef HAVE_HALF_MATH
                this->tmp_hp,
                this->tmp_perm_hp,
                this->x_hp,
#endif
#endif
#endif
                n_rows
            );
        }
        else if(solver_type == "gmres"){
            gmres_iteration_ref_cpu<VT>(
                sparse_mat, 
                timers,
                this->preconditioner_type,
                this->gmres_args->beta,
                this->D,
                this->gmres_args->V,
                this->gmres_args->H,
                this->gmres_args->H_tmp,
                this->gmres_args->J,
                this->gmres_args->Q,
                this->gmres_args->Q_copy,
                this->tmp, 
                this->tmp_perm, 
                this->gmres_args->R,
                this->gmres_args->g,
                this->gmres_args->g_copy,
                this->b, 
                this->x,
#ifdef USE_USPMV
#ifdef USE_AP
                this->gmres_args->V_dp,
                this->tmp_dp,
                this->tmp_perm_dp,
                this->gmres_args->V_sp,
                this->tmp_sp,
                this->tmp_perm_sp,
#ifdef HAVE_HALF_MATH
                this->gmres_args->V_hp,
                this->tmp_hp,
                this->tmp_perm_hp,
#endif
#endif
#endif
                n_rows,
                this->gmres_args->restart_count,
                iter_count,
                residual_norm,
                this->gmres_args->restart_length
            );
        }
        else if(solver_type == "conjugate-gradient"){
            cg_iteration_ref_cpu<VT>(
                sparse_mat,
                timers,
                this->preconditioner_type,
                &this->cg_args->alpha,
                &this->cg_args->beta,
                this->r,
                this->x_old,
                this->x_new,
                this->tmp,
                this->tmp_perm,
                this->cg_args->p_old,
                this->cg_args->p_new,
                this->cg_args->r_old,
                this->cg_args->r_new,
#ifdef USE_USPMV
#ifdef USE_AP
                this->tmp_dp,
                this->tmp_perm_dp,
                this->cg_args->p_old_dp,
                this->cg_args->p_new_dp,
                this->tmp_sp,
                this->tmp_perm_sp,
                this->cg_args->p_old_sp,
                this->cg_args->p_new_sp,
#ifdef HAVE_HALF_MATH
                this->tmp_hp,
                this->tmp_perm_hp,
                this->cg_args->p_old_hp,
                this->cg_args->p_new_hp,
#endif
#endif
#endif
                n_rows
            );
            std::swap(this->r, this->cg_args->r_new);
            // for(int i = 0; i < n_rows; ++i){
            //     this->r[i] = this->cg_args->r_new[i];
            // }
        }
    }

    void restart_gmres(
        Timers *timers,
        SparseMtxFormat<VT> *sparse_mat,
        int vec_size,
        int n_cols,
        int iter_count
    ){
#ifdef DEBUG_MODE
        std::cout << "RESTART GMRES" << std::endl;
#endif
        timers->gmres_get_x_wtime->start_stopwatch();
            gmres_get_x<VT>(
                this->gmres_args->R, 
                this->gmres_args->g, 
                this->x, 
                this->x_old, 
                this->gmres_args->V, 
                this->gmres_args->Vy, 
#ifdef USE_AP
                this->x_dp, 
                this->x_old_dp, 
                this->gmres_args->V_dp, 
                this->gmres_args->Vy_dp, 
                this->x_sp, 
                this->x_old_sp, 
                this->gmres_args->V_sp, 
                this->gmres_args->Vy_sp, 
#ifdef HAVE_HALF_MATH
                this->x_hp, 
                this->x_old_hp, 
                this->gmres_args->V_hp, 
                this->gmres_args->Vy_hp, 
#endif
#endif
                n_cols, 
                this->gmres_args->restart_count, 
                iter_count, 
                this->gmres_args->restart_length
            );
        timers->gmres_get_x_wtime->end_stopwatch();

#ifdef USE_AP
        if(xstr(WORKING_PRECISION) == "double"){
            calc_residual_cpu<VT,double>(sparse_mat, this->x_dp, this->b, this->r, this->tmp_dp, this->tmp_perm_dp, n_cols);
            apply_residual_preconditioner<double>(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);
        }
        else if(xstr(WORKING_PRECISION) == "float"){
            calc_residual_cpu<VT,float>(sparse_mat, this->x_sp, this->b, this->r, this->tmp_sp, this->tmp_perm_sp, n_cols);
            apply_residual_preconditioner<float>(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);
        }
        else if(xstr(WORKING_PRECISION) == "half"){
#ifdef HAVE_HALF_MATH
            calc_residual_cpu<VT,_Float16>(sparse_mat, this->x_hp, this->b, this->r, this->tmp_hp, this->tmp_perm_hp, n_cols);
            apply_residual_preconditioner<_Float16>(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);
#endif
        }
#else
        calc_residual_cpu<VT,VT>(sparse_mat, this->x, this->b, this->r, this->tmp, this->tmp_perm, n_cols);
        apply_residual_preconditioner<VT>(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);
#endif

    #ifdef DEBUG_MODE
        printf("restart residual = [");
        for(int i = 0; i < n_cols; ++i){
            std::cout << static_cast<double>(this->r[i]) << ",";
        }
        printf("]\n");
    #endif

        this->gmres_args->beta = euclidean_vec_norm_cpu(this->r, vec_size); 
        scale_residual<VT>(this->gmres_args->init_v, this->r, 1 / this->gmres_args->beta, vec_size);

    #ifdef DEBUG_MODE
        std::cout << "Restarted Beta = " << this->gmres_args->beta << std::endl;          

        // TODO: Is this vec_size, or n_cols?
        std::cout << "init_v = [";
            for(int i = 0; i < vec_size; ++i){
                std::cout << static_cast<double>(this->gmres_args->init_v[i]) << ", ";
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
            std::cout << static_cast<double>(this->x[i]) << std::endl;
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
        if(solver_type == "jacobi" || solver_type == "conjugate-gradient"){
            // TODO: Bandaid, since swapping seems to not work :(
#ifdef USE_AP
        std::string working_precision = xstr(WORKING_PRECISION);
        if(working_precision == "double"){
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star_dp[i] = this->x_old_dp[i];
            }
        }
        else if(working_precision == "float"){
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star_sp[i] = this->x_old_sp[i];
            }
        }
        else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star_hp[i] = this->x_old_hp[i];
            }
#endif
        }
#else
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star[i] = this->x_old[i];
#ifdef DEBUG_MODE
                std::cout << "args->x_star[" << i << "] = " << static_cast<double>(this->x_old[i]) << "==" << static_cast<double>(this->x_star[i]) << std::endl;
#endif
            }
            // std::swap(args->x_star, x_old);
#endif

        }
        else if (solver_type == "gauss-seidel"){
            #pragma omp parallel for
            for(int i = 0; i < vec_size; ++i){
                this->x_star[i] = this->x[i];
#ifdef DEBUG_MODE
                std::cout << "args->x_star[" << i << "] = " << static_cast<double>(this->x[i]) << "==" << static_cast<double>(this->x_star[i]) << std::endl;
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

    void print_x(
        int vec_size,
        int n_rows,
        int iter_count
    ){
        if(solver_type == "jacobi"){
            iter_output<VT>(this->x_new, vec_size, iter_count);
        }
        if(solver_type == "gauss-seidel" || solver_type == "conjugate-gradient"){
            iter_output<VT>(this->x, vec_size, iter_count);
        }
        if(solver_type == "gmres"){
            gmres_get_x<VT>(
                this->gmres_args->R, 
                this->gmres_args->g, 
                this->x, 
                this->x_old, 
                this->gmres_args->V, 
                this->gmres_args->Vy, 
#ifdef USE_AP
                this->x_dp, 
                this->x_old_dp, 
                this->gmres_args->V_dp, 
                this->gmres_args->Vy_dp, 
                this->x_sp, 
                this->x_old_sp, 
                this->gmres_args->V_sp, 
                this->gmres_args->Vy_sp, 
#ifdef HAVE_HALF_MATH
                this->x_hp, 
                this->x_old_hp, 
                this->gmres_args->V_hp, 
                this->gmres_args->Vy_hp, 
#endif
#endif
                n_rows, 
                this->gmres_args->restart_count, 
                iter_count, 
                this->gmres_args->restart_length
            );
            iter_output<VT>(this->x, vec_size, iter_count);
        }
    }

    void exchange_arrays(int vec_size){
        if(solver_type == "jacobi"){
        // TODO: Just swap pointers!
#ifdef USE_AP
    std::string working_precision = xstr(WORKING_PRECISION);
    if(working_precision == "double"){
        for(int i = 0; i < vec_size; ++i){
            this->x_old_dp[i] = this->x_new_dp[i];
        }
    }
    else if(working_precision == "float"){
        for(int i = 0; i < vec_size; ++i){
            this->x_old_sp[i] = this->x_new_sp[i];
        }
    }
    else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
        for(int i = 0; i < vec_size; ++i){
            this->x_old_hp[i] = this->x_new_hp[i];
        }
#endif
    }

#else
            for(int i = 0; i < vec_size; ++i){
                this->x_old[i] = this->x_new[i];
            }
#endif
        }
        else if(solver_type == "conjugate-gradient"){
#ifdef USE_AP
        // TODO: Just for testing, will probably need to figure out a better scheme
        std::string working_precision = xstr(WORKING_PRECISION);
        if(working_precision == "double"){
            std::swap(this->cg_args->p_old_dp, this->cg_args->p_new_dp);
            // #pragma omp parallel for
            // for(int i = 0; i < vec_size; ++i){
            //     this->cg_args->p_old_dp[i] = static_cast<double>(this->cg_args->p_new[i]);
            // }
        }
        else if(working_precision == "float"){
            std::swap(this->cg_args->p_old_sp, this->cg_args->p_new_sp);
            // #pragma omp parallel for
            // for(int i = 0; i < vec_size; ++i){
            //     this->cg_args->p_old_sp[i] = static_cast<float>(this->cg_args->p_new[i]);
            // }
        }
        else if(working_precision == "half"){
#ifdef HAVE_HALF_MATH
            std::swap(this->cg_args->p_old_hp, this->cg_args->p_new_hp);
            // #pragma omp parallel for
            // for(int i = 0; i < vec_size; ++i){
            //     this->cg_args->p_old_hp[i] = static_cast<_Float16>(this->cg_args->p_new[i]);
            // }
#endif
        }
#else
            std::swap(this->cg_args->p_old, this->cg_args->p_new);
#endif
            // Always happens regardless of USPMV
            // std::swap(this->cg_args->r_old, this->cg_args->r_new);
            std::swap(this->cg_args->r_old, this->r); // <- swapped r and r_new earlier
            std::swap(this->x_old, this->x_new);
        }
    }

    void init_residual(SparseMtxFormat<VT> *sparse_mat, int n_cols, int vec_size){
#ifdef USE_USPMV
        uspmv_scs_cpu<VT, VT, int>(
            sparse_mat->scs_mat->C,
            sparse_mat->scs_mat->n_chunks,
            &(sparse_mat->scs_mat->chunk_ptrs)[0],
            &(sparse_mat->scs_mat->chunk_lengths)[0],
            &(sparse_mat->scs_mat->col_idxs)[0],
            &(sparse_mat->scs_mat->values)[0],
            this->x_old,
            this->tmp_perm
        );
        apply_permutation<VT, int>(this->tmp, this->tmp_perm, &(sparse_mat->scs_mat->old_to_new_idx)[0], n_cols);
#else
        spmv_crs_cpu<VT>(this->tmp, sparse_mat->crs_mat, this->x_old);
#endif
        subtract_residual_cpu(this->r, this->b, this->tmp, n_cols);

        // Copy residual to AP structs
// #ifdef USE_AP
//         std::string working_precision = xstr(WORKING_PRECISION);
//         if(working_precision == "double"){
//             for(int i = 0; i < vec_size; ++i){
//                 this->r_dp[i] = this->r[i];
//             }
//             std::cout << "Before preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r_dp, n_cols)) << std::endl;
//         }
//         else if(working_precision == "float"){
//             for(int i = 0; i < vec_size; ++i){
//                 this->r_sp[i] = this->r[i];
//             }
//             std::cout << "Before preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r_sp, n_cols)) << std::endl;

//         }
//         else if(working_precision == "half"){
// #ifdef HAVE_HALF_MATH
//             for(int i = 0; i < vec_size; ++i){
//                 this->r_hp[i] = static_cast<_Float16>(this->r[i]); //????
//             }
//             std::cout << "Before preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r_hp, n_cols)) << std::endl;
// #endif
//         }
// #else
//             std::cout << "Before preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r, n_cols)) << std::endl;
// #endif

        // std::cout << "Before preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r, n_cols)) << std::endl;
        // std::cout << "r = [";
        // for(int i = 0; i < n_cols; ++i){
        //     std::cout << this->r[i] << ", ";
        // }
        // std::cout << "]" << std::endl;
        // std::vector<VT> res_copy(n_cols);
        // for(int i = 0; i < n_cols; ++i){
        //     res_copy[i] = this->r[i];
        // }
        // apply_residual_preconditioner(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);
        // apply_residual_preconditioner(this->preconditioner_type, sparse_mat, this->r, this->r, this->D, n_cols);

        // spltsv_crs<VT>(sparse_mat->crs_L, this->r, D, this->r);
    //     double sum;
    //     for(int row_idx = 0; row_idx < sparse_mat->crs_L->n_rows; ++row_idx){
    //         sum = 0.0;
    //         for(int nz_idx = sparse_mat->crs_L->row_ptr[row_idx]; nz_idx < sparse_mat->crs_L->row_ptr[row_idx+1]; ++nz_idx){
    //             sum += sparse_mat->crs_L->val[nz_idx] * this->r[sparse_mat->crs_L->col[nz_idx]];
    // // #ifdef DEBUG_MODE_FINE
    //             std::cout << sparse_mat->crs_L->val[nz_idx] << " * " << this->r[sparse_mat->crs_L->col[nz_idx]] << " = " << sparse_mat->crs_L->val[nz_idx] * this->r[sparse_mat->crs_L->col[nz_idx]] << " at idx: " << row_idx << std::endl; 
    // // #endif
    //         }
    //         std::cout << this->r[row_idx] << " - " << sum << " / " << D[row_idx] << " = " << (this->r[row_idx] - sum)/D[row_idx] << " at idx: " << row_idx << std::endl; 
    //         this->r[row_idx] = (this->r[row_idx] - sum)/D[row_idx];
    // // #ifdef DEBUG_MODE_FINE
    //         // std::cout << this->r[row_idx] << " - " << sum << " / " << D[row_idx] << " = " << this->r[row_idx] << " at idx: " << row_idx << std::endl; 
    // // #endif
    //     }
        // std::cout << "After preconditioning, residual 2-norm = " << static_cast<double>(euclidean_vec_norm_cpu(this->r, n_cols)) << std::endl;
        // std::cout << "r = [";
        // for(int i = 0; i < n_cols; ++i){
        //     std::cout << this->r[i] << ", ";
        // }
        // std::cout << "]" << std::endl;
        // exit(0);
    }

#ifdef USE_USPMV
    void unpermute_x_star(
        int n_cols,
        int *old_to_new_idx
    ){
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[n_cols];
    apply_permutation(x_star_perm, this->x_star, old_to_new_idx, n_cols);

    // Deep copy, so you can free memory
    // NOTE: You do not take SCS padding with you! 
    #pragma omp parallel for
    for(int i = 0; i < n_cols; ++i){
        this->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
    }

    void permute_arrays(
        int vec_size,
        int *old_to_new_idx,
        int *new_to_old_idx
    ){
        // Permute these vectors in accordance with SIGMA if using USpMV library
        double *D_perm = new double [vec_size];
        apply_permutation(D_perm, this->D, old_to_new_idx, vec_size);
        // std::swap(D_perm, args->D);

        double *b_perm = new double [vec_size];
        apply_permutation(b_perm, this->b, old_to_new_idx, vec_size);
        // std::swap(b_perm, args->b);

        // NOTE: Permuted w.r.t. columns due to symmetric permutation
        // double *x_old_perm = new double[vec_size];
        // apply_permutation(x_old_perm, this->x_old, new_to_old_idx, vec_size);
        // std::swap(x_old_perm, args->x_old);

        // Deep copy, so you can free memory
        // TODO: wrap in func
        for(int i = 0; i < vec_size; ++i){
            this->D[i] = D_perm[i]; // ?? Double Check!
            this->b[i] = b_perm[i]; // ?? Double Check!
            // this->x_old[i] = x_old_perm[i];
        }

        delete D_perm;
        delete b_perm;
        // delete x_old_perm;
    }  
#endif

};
#endif