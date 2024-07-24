#include "kernels.hpp"
#include "utility_funcs.hpp"
#include "io_funcs.hpp"
#include "methods/jacobi.hpp"
#include "methods/gauss_seidel.hpp"
#include "methods/gmres.hpp"

#ifdef USE_LIKWID
#include <likwid-marker.h>
#endif

void solve_cpu_rf(
    argType *args
){
    std::cout << "Entering Solver Harness" << std::endl;

    // Do we need fresh x anymore?
    double *x = new double[args->vec_size];
    double *x_new = new double[args->vec_size];
    double *x_old = new double[args->vec_size];
    // SparseMtxFormat *sparse_mat = args->sparse_mat;
    // Flags *flags = args->flags;
    // Solver *solver = args->solver;
    double residual_norm;

    args->solver->copy_fresh_x(x, x_new, x_old, args->vec_size);

    if(args->flags->print_iters){
        iter_output(args->solver->x_old, args->vec_size, args->loop_params->iter_count);
        printf("\n");
    }

    if(args->solver_type == "gmres"){
        init_gmres_structs(args->solver->gmres_args, args->vec_size);
        init_gmres_timers(args);
    }

#ifdef DEBUG_MODE
    std::cout << "x vector:" << std::endl;
    for(int i = 0; i < args->vec_size; ++i){
        std::cout << x[i] << std::endl;
    }
#endif

#ifdef USE_LIKWID
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
#endif


    do{
#ifdef DEBUG_MODE
        if(args->solver_type == "gmres"){
            std::cout << "Restarted GMRES inputting the x vector [" << std::endl;
            for(int i = 0; i < args->vec_size; ++i){
                std::cout << x[i] << std::endl;
            }
            std::cout << "]" << std::endl;
        }
#endif

        args->timers->solver_wtime->start_stopwatch();
        // TODO: change to solver class methods
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            jacobi_iteration_ref_cpu(
                args->sparse_mat, 
                args->solver->D, 
                args->solver->b, 
                args->solver->x_old, 
                args->solver->x_new
            );
            // jacobi_iteration_sep_cpu(sparse_mat, D, b, x_old, x_new, args->vec_size);
        }
        else if(args->solver_type == "gauss-seidel"){
            // For a reference solution, not meant for use with USpMV library
            gs_iteration_ref_cpu(
                args->sparse_mat, 
                args->solver->tmp, 
                args->solver->D, 
                args->solver->b, 
                args->solver->x
            );
            // gs_iteration_sep_cpu(sparse_mat, tmp, D, b, x, args->vec_size);
        }
        else if(args->solver_type == "gmres"){
            gmres_iteration_ref_cpu(
                args->sparse_mat, 
                args->timers,
                args->solver->gmres_args->V,
                args->solver->gmres_args->H,
                args->solver->gmres_args->H_tmp,
                args->solver->gmres_args->J,
                args->solver->gmres_args->Q,
                args->solver->gmres_args->Q_copy,
                args->solver->tmp, 
                args->solver->gmres_args->R,
                args->solver->gmres_args->g,
                args->solver->gmres_args->g_copy,
                args->solver->b, 
                args->solver->x,
                args->solver->gmres_args->beta,
                args->vec_size,
                args->solver->gmres_args->restart_count,
                args->loop_params->iter_count,
                &residual_norm,
                args->loop_params->gmres_restart_len
            );
        }
        args->timers->solver_wtime->end_stopwatch();

        // Record residual every "residual_check_len" iterations
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            // if(args->solver_type == "gmres"){
            //     args->timers->gmres_get_x_wtime->start_stopwatch();
            //     gmres_get_x(args->R, args->g, x, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count, args->loop_params->gmres_restart_len);
            //     args->timers->gmres_get_x_wtime->end_stopwatch();
            // }
            record_residual_norm(args, args->flags, args->sparse_mat, &residual_norm, args->solver->r, args->solver->x, args->solver->b, args->solver->x_new, args->solver->tmp);
        }

        if(args->flags->print_iters)
            print_x(args, args->solver->x, args->solver->x_new, args->solver->x_old, args->vec_size);  

        // Need to swap arrays every iteration in the case of Jacobi solver
        if(args->solver_type == "jacobi"){
            for(int i = 0; i < args->vec_size; ++i){
                args->solver->x_old[i] = args->solver->x_new[i];
            }
        }
            // std::swap(args->solver->x_new, args->solver->x_old);

#ifdef DEBUG_MODE
        std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
#endif 
        if(args->solver_type == "gmres"){
            // Decide to restart or not 
            if(residual_norm <= args->loop_params->stopping_criteria || \
                args->loop_params->iter_count >= args->loop_params->max_iters){
                    
            }
            else if ( (args->loop_params->iter_count+1) % args->solver->gmres_args->gmres_restart_len == 0 ){
                // Restart GMRES
#ifdef DEBUG_MODE
                std::cout << "RESTART GMRES" << std::endl;
#endif
                args->timers->gmres_get_x_wtime->start_stopwatch();
                gmres_get_x(
                    args->solver->gmres_args->R, 
                    args->solver->gmres_args->g, 
                    args->solver->x, 
                    args->solver->x_old, 
                    args->solver->gmres_args->V, 
                    args->solver->gmres_args->Vy, 
                    args->coo_mat->n_cols, 
                    args->solver->gmres_args->restart_count, 
                    args->loop_params->iter_count, 
                    args->loop_params->gmres_restart_len
                );
                args->timers->gmres_get_x_wtime->end_stopwatch();
                calc_residual_cpu(args->sparse_mat, args->solver->x, args->solver->b, args->solver->r, args->solver->tmp, args->coo_mat->n_cols);
#ifdef DEBUG_MODE
                printf("restart residual = [");
                for(int i = 0; i < args->coo_mat->n_cols; ++i){
                    std::cout << args->solver->r[i] << ",";
                }
                printf("]\n");
#endif
                args->solver->gmres_args->beta = euclidean_vec_norm_cpu(args->solver->r, args->coo_mat->n_cols); 
                scale(args->solver->gmres_args->init_v, args->solver->r, 1 / args->solver->gmres_args->beta, args->coo_mat->n_cols);
    
#ifdef DEBUG_MODE
                std::cout << "Restarted Beta = " << args->solver->gmres_args->beta << std::endl;          

                std::cout << "init_v = [";
                    for(int i = 0; i < args->vec_size; ++i){
                        std::cout << args->solver->gmres_args->init_v[i] << ", ";
                    }
                std::cout << "]" << std::endl;
#endif
                double norm_r0 = euclidean_vec_norm_cpu(args->solver->r, args->vec_size);

#ifdef DEBUG_MODE
                printf("restarted norm(initial residual) = %f\n", norm_r0);
#endif
                init_gmres_structs(args->solver->gmres_args, args->vec_size);
                ++args->solver->gmres_args->restart_count;
#ifdef DEBUG_MODE
                std::cout << "Restarted GMRES outputting the x vector [" << std::endl;
                for(int i = 0; i < args->vec_size; ++i){
                    std::cout << args->solver->x[i] << std::endl;
                }
                std::cout << "]" << std::endl;
#endif

                // Need deep copy TODO: rethink and do better
                #pragma omp parallel for
                for(int i = 0; i < args->vec_size; ++i){
                    args->solver->x_old[i] = args->solver->x[i];
                }

            }
        }

        ++args->loop_params->iter_count;


        // std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;

    } while(residual_norm > args->loop_params->stopping_criteria && \
    args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    if(args->solver_type == "jacobi"){
        // TODO: Bandaid, sincne swapping seems to not work :(
        #pragma omp parallel for
        for(int i = 0; i < args->vec_size; ++i){
            args->solver->x_star[i] = args->solver->x_old[i];
            // std::cout << "args->x_star[" << i << "] = " << x_old[i] << "==" << args->x_star[i] << std::endl;
        }
        // std::swap(args->x_star, x_old);

    }
    else if (args->solver_type == "gauss-seidel"){
        #pragma omp parallel for
        for(int i = 0; i < args->vec_size; ++i){
            args->solver->x_star[i] = args->solver->x[i];
            // std::cout << "args->x_star[" << i << "] = " << x[i] << "==" << args->x_star[i] << std::endl;
        }
        // std::swap(x, args->x_star);
    }    
    
    // Record final residual with approximated solution vector x_star
    if(args->solver_type == "gmres"){
        // Only needed if you actually want the x vector
        // args->timers->gmres_get_x_wtime->start_stopwatch();
        // gmres_get_x(args->R, args->g, args->x_star, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count - 1, args->loop_params->gmres_restart_len);
        // args->timers->gmres_get_x_wtime->end_stopwatch();
    }

    record_residual_norm(
        args, 
        args->flags, 
        args->sparse_mat, 
        &residual_norm, 
        args->solver->r, 
        args->solver->x_star, 
        args->solver->b, 
        args->solver->x_star, 
        args->solver->tmp
    );

#ifdef USE_USPMV
    // Bring final result vector out of permuted space
    double *x_star_perm = new double[args->vec_size];
    apply_permutation(x_star_perm, args->solver->x_star, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);

    // Deep copy, so you can free memory
    // NOTE: You do not take SCS padding with you! 
    #pragma omp parallel for
    for(int i = 0; i < args->coo_mat->n_cols; ++i){
        args->solver->x_star[i] = x_star_perm[i];
    }

    delete x_star_perm;
#endif

    delete x;
    delete x_new;
    delete x_old;
}

// void solve_cpu(
//     argType *args
// ){
//     std::cout << "Entering Solver Harness" << std::endl;

//     double *x = new double[args->vec_size];
//     double *x_new = new double[args->vec_size];
//     double *x_old = new double[args->vec_size];
//     SparseMtxFormat *sparse_mat = args->sparse_mat;
//     Flags *flags = args->flags;
//     double residual_norm;

//     // Need deep copy TODO: rethink and do better
//     #pragma omp parallel for
//     for(int i = 0; i < args->vec_size; ++i){
//         x[i] = args->x_old[i];
//         x_new[i] = args->x_new[i];
//         x_old[i] = args->x_old[i];
//     }

//     if(args->flags->print_iters){
//         iter_output(x, args->vec_size, args->loop_params->iter_count);
//         printf("\n");
//     }

//     if(args->solver_type == "gmres"){
//         init_gmres_structs(args, args->vec_size);
//         init_gmres_timers(args);
//     }

// #ifdef DEBUG_MODE
//     std::cout << "x vector:" << std::endl;
//     for(int i = 0; i < args->vec_size; ++i){
//         std::cout << x[i] << std::endl;
//     }
// #endif

// #ifdef USE_LIKWID
//     #pragma omp parallel
//     {
// #ifdef USE_USPMV
// #ifdef USE_AP
//         LIKWID_MARKER_REGISTER("uspmv_ap_crs_benchmark");
// #else
//         LIKWID_MARKER_REGISTER("uspmv_crs_benchmark");
// #endif

// #else 
//         LIKWID_MARKER_REGISTER("native_spmv_benchmark");
// #endif
//     }
// #endif


//     do{
// #ifdef DEBUG_MODE
//         std::cout << "Restarted GMRES inputting the x vector [" << std::endl;
//         for(int i = 0; i < args->vec_size; ++i){
//             std::cout << x[i] << std::endl;
//         }
//         std::cout << "]" << std::endl;
// #endif

//         args->timers->solver_wtime->start_stopwatch();
//         // TODO: change to solver class methods
//         if(args->solver_type == "jacobi"){
//             // For a reference solution, not meant for use with USpMV library
//             jacobi_iteration_ref_cpu(
//                 args->sparse_mat, 
//                 args->D, 
//                 args->b, 
//                 x_old, 
//                 x_new
//             );
//             // jacobi_iteration_sep_cpu(sparse_mat, D, b, x_old, x_new, args->vec_size);
//         }
//         else if(args->solver_type == "gauss-seidel"){
//             // For a reference solution, not meant for use with USpMV library
//             gs_iteration_ref_cpu(
//                 args->sparse_mat, 
//                 args->tmp, 
//                 args->D, 
//                 args->b, 
//                 x
//             );
//             // gs_iteration_sep_cpu(sparse_mat, tmp, D, b, x, args->vec_size);
//         }
//         else if(args->solver_type == "gmres"){
//             gmres_iteration_ref_cpu(
//                 sparse_mat, 
//                 args->timers,
//                 args->V,
//                 args->H,
//                 args->H_tmp,
//                 args->J,
//                 args->Q,
//                 args->Q_copy,
//                 args->tmp, 
//                 args->R,
//                 args->g,
//                 args->g_copy,
//                 args->b, 
//                 x,
//                 args->beta,
//                 args->vec_size,
//                 args->restart_count,
//                 args->loop_params->iter_count,
//                 &residual_norm,
//                 args->loop_params->gmres_restart_len
//             );
//         }
//         args->timers->solver_wtime->end_stopwatch();

//         // Record residual every "residual_check_len" iterations
//         if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
//             // if(args->solver_type == "gmres"){
//             //     args->timers->gmres_get_x_wtime->start_stopwatch();
//             //     gmres_get_x(args->R, args->g, x, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count, args->loop_params->gmres_restart_len);
//             //     args->timers->gmres_get_x_wtime->end_stopwatch();
//             // }
//             record_residual_norm(args, flags, sparse_mat, &residual_norm, args->r, x, args->b, x_new, args->tmp);
//         }

//         if(flags->print_iters)
//             print_x(args, x, x_new, x_old, args->vec_size);  

//         // Need to swap arrays every iteration in the case of Jacobi solver
//         if(args->solver_type == "jacobi")
//             std::swap(x_new, x_old);

// #ifdef DEBUG_MODE
//         std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
// #endif 
//         if(args->solver_type == "gmres"){
//             // Decide to restart or not 
//             if(residual_norm <= args->loop_params->stopping_criteria || \
//                 args->loop_params->iter_count >= args->loop_params->max_iters){
                    
//             }
//             else if ( (args->loop_params->iter_count+1) % args->gmres_restart_len == 0 ){
//                 // Restart GMRES
// #ifdef DEBUG_MODE
//                 std::cout << "RESTART GMRES" << std::endl;
// #endif
//                 args->timers->gmres_get_x_wtime->start_stopwatch();
//                 gmres_get_x(args->R, args->g, x, x_old, args->V, args->Vy, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count, args->loop_params->gmres_restart_len);
//                 args->timers->gmres_get_x_wtime->end_stopwatch();
//                 calc_residual_cpu(args->sparse_mat, x, args->b, args->r, args->tmp, args->coo_mat->n_cols);
// #ifdef DEBUG_MODE
//                 printf("restart residual = [");
//                 for(int i = 0; i < args->coo_mat->n_cols; ++i){
//                     std::cout << args->r[i] << ",";
//                 }
//                 printf("]\n");
// #endif
//                 args->beta = euclidean_vec_norm_cpu(args->r, args->coo_mat->n_cols); 
//                 scale(args->init_v, args->r, 1 / args->beta, args->coo_mat->n_cols);
    
// #ifdef DEBUG_MODE
//                 std::cout << "Restarted Beta = " << args->beta << std::endl;          

//                 std::cout << "init_v = [";
//                     for(int i = 0; i < args->vec_size; ++i){
//                         std::cout << args->init_v[i] << ", ";
//                     }
//                 std::cout << "]" << std::endl;
// #endif
//                 double norm_r0 = euclidean_vec_norm_cpu(args->r, args->vec_size);

// #ifdef DEBUG_MODE
//                 printf("restarted norm(initial residual) = %f\n", norm_r0);
// #endif
//                 init_gmres_structs(args, args->vec_size);
//                 ++args->restart_count;
// #ifdef DEBUG_MODE
//                 std::cout << "Restarted GMRES outputting the x vector [" << std::endl;
//                 for(int i = 0; i < args->vec_size; ++i){
//                     std::cout << x[i] << std::endl;
//                 }
//                 std::cout << "]" << std::endl;
// #endif

//                 // Need deep copy TODO: rethink and do better
//                 #pragma omp parallel for
//                 for(int i = 0; i < args->vec_size; ++i){
//                     x_old[i] = x[i];
//                 }

//             }
//         }

//         ++args->loop_params->iter_count;


//         // std::cout << residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;

//     } while(residual_norm > args->loop_params->stopping_criteria && \
//     args->loop_params->iter_count < args->loop_params->max_iters);

//     args->flags->convergence_flag = (residual_norm <= args->loop_params->stopping_criteria) ? true : false;

//     if(args->solver_type == "jacobi"){
//         // TODO: Bandaid, sincne swapping seems to not work :(
//         #pragma omp parallel for
//         for(int i = 0; i < args->vec_size; ++i){
//             args->x_star[i] = x_old[i];
//             // std::cout << "args->x_star[" << i << "] = " << x_old[i] << "==" << args->x_star[i] << std::endl;
//         }
//         // std::swap(args->x_star, x_old);

//     }
//     else if (args->solver_type == "gauss-seidel"){
//         #pragma omp parallel for
//         for(int i = 0; i < args->vec_size; ++i){
//             args->x_star[i] = x[i];
//             // std::cout << "args->x_star[" << i << "] = " << x[i] << "==" << args->x_star[i] << std::endl;
//         }
//         // std::swap(x, args->x_star);
//     }    
    
//     // Record final residual with approximated solution vector x_star
//     if(args->solver_type == "gmres"){
//         // Only needed if you actually want the x vector
//         // args->timers->gmres_get_x_wtime->start_stopwatch();
//         // gmres_get_x(args->R, args->g, args->x_star, x_old, args->V, args->coo_mat->n_cols, args->restart_count, args->loop_params->iter_count - 1, args->loop_params->gmres_restart_len);
//         // args->timers->gmres_get_x_wtime->end_stopwatch();
//     }

//     record_residual_norm(args, flags, sparse_mat, &residual_norm, args->r, args->x_star, args->b, args->x_star, args->tmp);

// #ifdef USE_USPMV
//     // Bring final result vector out of permuted space
//     double *x_star_perm = new double[args->vec_size];
//     apply_permutation(x_star_perm, args->x_star, &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);

//     // Deep copy, so you can free memory
//     // NOTE: You do not take SCS padding with you! 
//     #pragma omp parallel for
//     for(int i = 0; i < args->coo_mat->n_cols; ++i){
//         args->x_star[i] = x_star_perm[i];
//     }

//     delete x_star_perm;
// #endif

//     delete x;
//     delete x_new;
//     delete x_old;
// }

#ifdef __CUDACC__
void solve_gpu(
    argType *args
){

    // NOTE: Only for convenience. Will change to UM later.
    double *h_residual_norm = new double;

    // TODO: Why does this get messed up? 
    args->loop_params->residual_count = 0;

    // // Unpack relevant args
    // double *d_x = args->d_x_old; // GS
    // double *d_x_new = args->d_x_new; // Jacobi
    // double *d_x_old = args->d_x_old; // Jacobi
    // int d_n_rows = args->coo_mat->n_rows;

    // // TODO: collect into a struct
    // int *d_row_ptr = args->d_row_ptr;
    // int *d_col = args->d_col;
    // double *d_val = args->d_val; 

    // double *d_tmp = args->d_tmp;
    // double *d_D = args->d_D;
    // double *d_r = args->d_r;
    // double *d_b = args->d_b;

    double *d_residual_norm;
    cudaMalloc(&d_residual_norm, sizeof(double));
    cudaMemset(d_residual_norm, 0.0, sizeof(double));

    Flags *flags = args->flags; 

    double residual_norm;

    // TODO: Adapt for GPUs
    // if(args->flags->print_iters){
    //     iter_output(d_x, args->loop_params->iter_count);
    //     printf("\n");
    // }

    // TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//     std::cout << "x vector:" << std::endl;
//     for(int i = 0; i < args->vec_size; ++i){
//         std::cout << d_x[i] << std::endl;
//     }
// #endif

    // Begin timer
    struct timeval calc_time_start, calc_time_end;
    start_time(&calc_time_start);

    do{
        if(args->solver_type == "jacobi"){
            // For a reference solution, not meant for use with USpMV library
            // jacobi_iteration_ref_gpu<<<BLOCKS_PER_GRID,THREADS_PER_BLOCK>>>(args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new, args->vec_size);
            jacobi_iteration_sep_gpu(args->vec_size, args->d_row_ptr, args->d_col, args->d_val, args->d_D, args->d_b, args->d_x_old, args->d_x_new);
        }
        else if(args->solver_type == "gauss-seidel"){
            // TODO: Adapt for GPUs
            printf("GS_solve still under development for GPUs.\n");
            exit(1);
            // For a reference solution, not meant for use with USpMV library
            // gs_iteration_ref_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
            // gs_iteration_sep_gpu(d_row_ptr, d_col, d_val, d_D, d_b, d_x_old, d_x_new);
        }
        
        if (args->loop_params->iter_count % args->loop_params->residual_check_len == 0){
            
            // Record residual every "residual_check_len" iterations
            if(args->solver_type == "jacobi"){
                calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_new, args->d_b, args->d_r, args->d_tmp, args->vec_size);
            }
            else if(args->solver_type == "gauss-seidel"){
                // TODO: Adapt for GPUs
                printf("GS_solve still under development for GPUs.\n");
                exit(1);
                // calc_residual_gpu(sparse_mat, x, b, r, tmp);
            }
            
///////////////////////////////////// Grrr DEBUG! //////////////////////////////////////////
            // For now, have to do this on the CPU. Giving up on GPU implementation
            // cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
            // *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);
            // TODO: Correct grid + block size?
            // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
            // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
            infty_vec_norm_gpu<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, THREADS_PER_BLOCK*sizeof(double)>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////// DEBUG! //////////////////////////////////////////

            // TODO: Put residual_norm in unified memory to avoid this transfer
            // NOTE: need to convert *double to *void
            cudaMemcpy(&(*h_residual_norm), &(*d_residual_norm), sizeof(double), cudaMemcpyDeviceToHost);
            // cudaDeviceSynchronize(); // <- not necessary
            // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

            // std::cout << "the first h_residual_norm = " << *h_residual_norm << std::endl;
            // exit(0);
            
            args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;
            ++args->loop_params->residual_count;

// TODO: Adapt for GPUs
//             if(flags->print_iters){
//                 if(args->solver_type == "jacobi"){
//                     iter_output(x_new, args->loop_params->iter_count);
//                 }
//                 else if(args->solver_type == "gauss-seidel"){
//                     iter_output(x, args->loop_params->iter_count);
//                 }
//             }
        }

// TODO: Adapt for GPUs
// #ifdef DEBUG_MODE
//         std::cout << "[";
//         if(args->solver_type == "jacobi"){
//             for(int i = 0; i < x_new->size(); ++i){
//                 std::cout << (*x_new)[i] << ", ";
//             }
//         }
//         else if (args->solver_type == "gauss-seidel"){
//             for(int i = 0; i < x->size(); ++i){
//                 std::cout << (*x)[i] << ", ";
//             }
//         }
//         std::cout << "]" << std::endl;
  
//         std::cout << "residual norm: " << infty_vec_norm(r) << std::endl;
//         std::cout << "stopping_criteria: " << args->loop_params->stopping_criteria << std::endl; 
// #endif  

// TODO: Adapt for GPUs???
        cudaDeviceSynchronize();
        if(args->solver_type == "jacobi"){
            // NOTE: Might work, might not..
            // TODO: huh?? Causes seg fault
            // std::cout << "d_x_new pointer: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer: " << d_x_old << std::endl;
            std::swap(args->d_x_new, args->d_x_old);
            // std::cout << "d_x_new pointer after swap: " << d_x_new << std::endl;
            // std::cout << "d_x_old pointer after swap: " << d_x_old << std::endl;
        }
    
        ++args->loop_params->iter_count;

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    // Do check on host for now, easiest
        // std::cout << *h_residual_norm << " <? " << args->loop_params->stopping_criteria << std::endl;
        // exit(0);
    } while(*h_residual_norm > args->loop_params->stopping_criteria && args->loop_params->iter_count < args->loop_params->max_iters);

    args->flags->convergence_flag = (*h_residual_norm <= args->loop_params->stopping_criteria) ? true : false;

    cudaDeviceSynchronize();
    if(args->solver_type == "jacobi"){
        // TODO: huh?? Causes seg fault
        std::swap(args->d_x_old, args->d_x_star);
    }
    else if (args->solver_type == "gauss-seidel"){
        // TODO: Adapt for GPUs
        printf("GS_solve still under development for GPUs.\n");
        exit(1);
        // std::swap(*x, *(args->x_star));
    }

    // Record final residual with approximated solution vector x
///////////////////////////////////// DEBUG! //////////////////////////////////////////
    // TODO: Giving up on GPU for this for now
    cudaMemcpy(args->r, args->d_r, args->vec_size * sizeof(double), cudaMemcpyDeviceToHost);
    *h_residual_norm = infty_vec_norm_cpu(args->r, args->vec_size);

    // calc_residual_gpu(args->d_row_ptr, args->d_col, args->d_val, args->d_x_star, args->d_r, args->d_b, args->d_tmp, args->vec_size);
    // infty_vec_norm_gpu<<<1,1>>>(args->d_r, d_residual_norm, args->vec_size);
///////////////////////////////////////////////////////////////////////////////////////

    // TODO: Put residual_norm in unified memory to avoid this transfer
    // cudaDeviceSynchronize();
    // cudaMemcpy(h_residual_norm, d_residual_norm, sizeof(double), cudaMemcpyDeviceToHost);

    args->normed_residuals[args->loop_params->residual_count] = *h_residual_norm;

// TODO: Adapt for GPUs
// #ifdef USE_USPMV
//     // Bring final result vector out of permuted space
//     std::vector<double> x_star_perm(args->vec_size, 0);
//     apply_permutation(&(x_star_perm)[0], &(*args->x_star)[0], &(args->sparse_mat->scs_mat->old_to_new_idx)[0], args->vec_size);
//     std::swap(x_star_perm, (*args->x_star));
// #endif

    // End timer
    args->calc_time_elapsed = end_time(&calc_time_start, &calc_time_end);

    // Why are you freeing this here?
    cudaFree(d_residual_norm);
    delete h_residual_norm;
}
#endif

void solve(
    argType *args
){
    timeval *solver_harness_time_start = new timeval;
    timeval *solver_harness_time_end = new timeval;
    Stopwatch *solver_harness_wtime = new Stopwatch(solver_harness_time_start, solver_harness_time_end);
    args->timers->solver_harness_wtime = solver_harness_wtime ;
    args->timers->solver_harness_wtime->start_stopwatch();

    timeval *solver_time_start = new timeval;
    timeval *solver_time_end = new timeval;
    Stopwatch *solver_wtime = new Stopwatch(solver_time_start, solver_time_end);
    args->timers->solver_wtime = solver_wtime ;

#ifndef __CUDACC__
    solve_cpu_rf(args);
#else
    solve_gpu(args);
#endif

    args->timers->solver_harness_wtime->end_stopwatch();
}